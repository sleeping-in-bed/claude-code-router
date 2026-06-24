import { createHash } from "crypto";
import { existsSync } from "fs";
import { writeFile } from "fs/promises";
import { Socket } from "net";
import { homedir } from "os";
import { join } from "path";
import { initConfig, initDir } from "./utils";
import { createServer } from "./server";
import { apiKeyAuth } from "./middleware/auth";
import { CONFIG_FILE, HOME_DIR, listPresets } from "@CCR/shared";
import { createStream } from 'rotating-file-stream';
import { sessionUsageCache } from "@musistudio/llms";
import { SSEParserTransform } from "./utils/SSEParser.transform";
import { SSESerializerTransform } from "./utils/SSESerializer.transform";
import { rewriteStream } from "./utils/rewriteStream";
import JSON5 from "json5";
import { IAgent, ITool } from "./agents/type";
import agentsManager from "./agents";
import { EventEmitter } from "node:events";
import { calculateTokenCount, pluginManager, tokenSpeedPlugin } from "@musistudio/llms";

const event = new EventEmitter()

async function initializeClaudeConfig() {
  const homeDir = homedir();
  const configPath = join(homeDir, ".claude.json");
  if (!existsSync(configPath)) {
    const userID = Array.from(
      { length: 64 },
      () => Math.random().toString(16)[2]
    ).join("");
    const configContent = {
      numStartups: 184,
      autoUpdaterStatus: "enabled",
      userID,
      hasCompletedOnboarding: true,
      lastOnboardingVersion: "1.0.17",
      projects: {},
    };
    await writeFile(configPath, JSON.stringify(configContent, null, 2));
  }
}

interface RunOptions {
  port?: number;
  logger?: any;
}

/**
 * Plugin configuration from config file
 */
interface PluginConfig {
  name: string;
  enabled?: boolean;
  options?: Record<string, any>;
}

type PromptCacheTTL = "5m" | "1h";

interface PromptCacheEntry {
  key: string;
  ttlType: PromptCacheTTL;
  prefixTokens: number;
  createdAt: number;
  expiresAt: number;
}

interface PromptCacheDescriptor {
  cacheKey: string;
  normalizedPrefixRequest: Record<string, any>;
  prefixHash: string;
  prefixRequest: Record<string, any>;
  ttlType: PromptCacheTTL;
}

interface PromptCacheSnapshot {
  request: Record<string, any>;
  ttlType: PromptCacheTTL;
}

interface PromptCacheCandidate extends PromptCacheDescriptor {
  snapshotIndex: number;
}

interface PromptCacheStoreCandidate {
  descriptor: PromptCacheDescriptor;
  prefixTokens: number;
}

interface ParsedRedisResponse {
  error?: Error;
  nextOffset: number;
  value: any;
}

interface PromptCacheSessionPointer {
  expiresAt: number;
  key: string;
  normalizedPrefixRequest: Record<string, any>;
  prefixHash: string;
  prefixTokens: number;
  ttlType: PromptCacheTTL;
  updatedAt: number;
}

interface PromptCacheUsage {
  inputTokens: number;
  cacheCreationInputTokens: number;
  cacheReadInputTokens: number;
  cacheCreation5mTokens: number;
  cacheCreation1hTokens: number;
  shouldStore: boolean;
  stored: boolean;
  descriptor?: PromptCacheDescriptor;
  sessionPointer?: PromptCacheSessionPointer;
  storeCandidates?: PromptCacheStoreCandidate[];
}

const PROMPT_CACHE_TTL_MS: Record<PromptCacheTTL, number> = {
  "5m": 5 * 60 * 1000,
  "1h": 60 * 60 * 1000,
};

class PromptCacheRedisStore {
  private readonly db: number;
  private readonly host: string;
  private readonly password: string;
  private readonly port: number;
  private readonly username: string;
  private buffer = Buffer.alloc(0);
  private commandChain = Promise.resolve();
  private connectPromise: Promise<void> | null = null;
  private pending: Array<{
    reject: (error: Error) => void;
    resolve: (value: any) => void;
  }> = [];
  private socket: Socket | null = null;

  constructor(redisUrl: string) {
    const parsed = new URL(redisUrl);
    if (parsed.protocol !== "redis:") {
      throw new Error(`unsupported prompt cache redis protocol: ${parsed.protocol}`);
    }

    this.host = parsed.hostname || "127.0.0.1";
    this.port = Number(parsed.port || "6379");
    this.username = decodeURIComponent(parsed.username || "");
    this.password = decodeURIComponent(parsed.password || "");
    this.db = Number(parsed.pathname.replace("/", "") || "0");
  }

  async get(cacheKey: string): Promise<PromptCacheEntry | null> {
    const parsed = await this.getJSON(cacheKey);
    if (!parsed || typeof parsed !== "object") {
      return null;
    }

    if (Number(parsed.expiresAt || 0) <= Date.now()) {
      await this.sendCommand(["DEL", cacheKey]);
      return null;
    }

    return {
      key: String(parsed.key || cacheKey),
      ttlType: parsed.ttlType === "1h" ? "1h" : "5m",
      prefixTokens: Math.max(0, Math.round(Number(parsed.prefixTokens || 0))),
      createdAt: Math.max(0, Math.round(Number(parsed.createdAt || 0))),
      expiresAt: Math.max(0, Math.round(Number(parsed.expiresAt || 0))),
    };
  }

  async ping(): Promise<void> {
    await this.sendCommand(["PING"]);
  }

  async getJSON(cacheKey: string): Promise<any | null> {
    const payload = await this.sendCommand(["GET", cacheKey]);
    if (payload === null) {
      return null;
    }

    return JSON.parse(String(payload));
  }

  async delete(cacheKey: string): Promise<void> {
    await this.sendCommand(["DEL", cacheKey]);
  }

  async setJSON(cacheKey: string, value: Record<string, any>, ttlMs: number): Promise<void> {
    await this.sendCommand([
      "SET",
      cacheKey,
      JSON.stringify(value),
      "PX",
      String(Math.max(1, ttlMs)),
    ]);
  }

  async setIfAbsent(entry: PromptCacheEntry): Promise<boolean> {
    const payload = JSON.stringify(entry);
    const ttlMs = Math.max(1, entry.expiresAt - Date.now());
    const result = await this.sendCommand([
      "SET",
      entry.key,
      payload,
      "PX",
      String(ttlMs),
      "NX",
    ]);
    return result === "OK";
  }

  private async connect(): Promise<void> {
    if (this.socket && !this.socket.destroyed) {
      return;
    }

    if (this.connectPromise) {
      return this.connectPromise;
    }

    this.connectPromise = new Promise<void>((resolve, reject) => {
      const socket = new Socket();

      const cleanupConnectHandlers = () => {
        socket.removeListener("connect", handleConnect);
        socket.removeListener("error", handleConnectError);
      };

      const handleConnect = async () => {
        cleanupConnectHandlers();
        this.socket = socket;
        this.socket.setNoDelay(true);
        this.socket.on("data", (chunk) => this.handleData(chunk));
        this.socket.on("close", () => this.handleDisconnect(new Error("prompt cache redis connection closed")));
        this.socket.on("error", (error) => this.handleDisconnect(error instanceof Error ? error : new Error(String(error))));

        try {
          if (this.password) {
            if (this.username) {
              await this.sendCommandDirect(["AUTH", this.username, this.password], true);
            } else {
              await this.sendCommandDirect(["AUTH", this.password], true);
            }
          }

          if (this.db > 0) {
            await this.sendCommandDirect(["SELECT", String(this.db)], true);
          }

          resolve();
        } catch (error) {
          socket.destroy();
          reject(error);
        }
      };

      const handleConnectError = (error: Error) => {
        cleanupConnectHandlers();
        reject(error);
      };

      socket.once("connect", handleConnect);
      socket.once("error", handleConnectError);
      socket.connect(this.port, this.host);
    }).finally(() => {
      this.connectPromise = null;
    });

    return this.connectPromise;
  }

  private encodeCommand(parts: string[]): Buffer {
    const chunks: string[] = [`*${parts.length}\r\n`];
    for (const part of parts) {
      const byteLength = Buffer.byteLength(part);
      chunks.push(`$${byteLength}\r\n${part}\r\n`);
    }
    return Buffer.from(chunks.join(""), "utf8");
  }

  private handleData(chunk: Buffer) {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    while (this.pending.length > 0) {
      const parsed = this.parseResponse(this.buffer, 0);
      if (!parsed) {
        return;
      }

      this.buffer = this.buffer.subarray(parsed.nextOffset);
      const pending = this.pending.shift();
      if (!pending) {
        return;
      }

      if (parsed.error) {
        pending.reject(parsed.error);
        continue;
      }

      pending.resolve(parsed.value);
    }
  }

  private handleDisconnect(error: Error) {
    this.socket = null;
    this.buffer = Buffer.alloc(0);
    while (this.pending.length > 0) {
      const pending = this.pending.shift();
      pending?.reject(error);
    }
  }

  private parseResponse(buffer: Buffer, offset: number): ParsedRedisResponse | null {
    if (buffer.length <= offset) {
      return null;
    }

    const prefix = String.fromCharCode(buffer[offset]);
    const lineEnd = buffer.indexOf("\r\n", offset);
    if (lineEnd === -1) {
      return null;
    }

    const line = buffer.toString("utf8", offset + 1, lineEnd);
    const cursor = lineEnd + 2;

    if (prefix === "+") {
      return { nextOffset: cursor, value: line };
    }

    if (prefix === "-") {
      return { error: new Error(line), nextOffset: cursor, value: null };
    }

    if (prefix === ":") {
      return { nextOffset: cursor, value: Number(line) };
    }

    if (prefix === "$") {
      const length = Number(line);
      if (length === -1) {
        return { nextOffset: cursor, value: null };
      }

      const end = cursor + length;
      if (buffer.length < end + 2) {
        return null;
      }

      return {
        nextOffset: end + 2,
        value: buffer.toString("utf8", cursor, end),
      };
    }

    if (prefix === "*") {
      const count = Number(line);
      if (count === -1) {
        return { nextOffset: cursor, value: null };
      }

      const values: any[] = [];
      let nextOffset = cursor;
      for (let index = 0; index < count; index += 1) {
        const item = this.parseResponse(buffer, nextOffset);
        if (!item) {
          return null;
        }
        if (item.error) {
          return item;
        }
        values.push(item.value);
        nextOffset = item.nextOffset;
      }

      return { nextOffset, value: values };
    }

    throw new Error(`unsupported redis response prefix: ${prefix}`);
  }

  private async sendCommand(parts: string[]): Promise<any> {
    const result = this.commandChain.then(() => this.sendCommandDirect(parts));
    this.commandChain = result.then(() => undefined, () => undefined);
    return result;
  }

  private async sendCommandDirect(parts: string[], assumeConnected = false): Promise<any> {
    if (!assumeConnected) {
      await this.connect();
    }
    const socket = this.socket;
    if (!socket || socket.destroyed) {
      throw new Error("prompt cache redis connection is not available");
    }

    return new Promise((resolve, reject) => {
      this.pending.push({ resolve, reject });
      socket.write(this.encodeCommand(parts));
    });
  }
}

let promptCacheStore: PromptCacheRedisStore | null = null;

function cloneJSON<T>(value: T): T {
  return JSON.parse(JSON.stringify(value));
}

function normalizePromptCacheRequest(value: Record<string, any>): Record<string, any> {
  const normalized = cloneJSON(value);

  const stripCacheControl = (input: any): any => {
    if (Array.isArray(input)) {
      return input.map((item) => stripCacheControl(item));
    }

    if (!input || typeof input !== "object") {
      return input;
    }

    const next: Record<string, any> = {};
    for (const [key, item] of Object.entries(input)) {
      if (key === "cache_control") {
        continue;
      }
      next[key] = stripCacheControl(item);
    }
    return next;
  };

  const normalizeMessageContent = (message: any) => {
    if (!message || typeof message !== "object") {
      return;
    }

    if (typeof message.content === "string") {
      return;
    }

    if (!Array.isArray(message.content) || message.content.length !== 1) {
      return;
    }

    const [part] = message.content;
    if (!part || typeof part !== "object" || part.type !== "text" || typeof part.text !== "string") {
      return;
    }

    const partKeys = Object.keys(part).filter((key) => key !== "cache_control");
    if (partKeys.length === 2 && partKeys.includes("type") && partKeys.includes("text")) {
      message.content = part.text;
    }
  };

  const normalizedWithoutCacheControl = stripCacheControl(normalized);

  if (Array.isArray(normalizedWithoutCacheControl.system)) {
    for (const block of normalizedWithoutCacheControl.system) {
      if (!block || typeof block !== "object" || typeof block.text !== "string") {
        continue;
      }

      if (!block.text.startsWith("x-anthropic-billing-header:")) {
        continue;
      }

      block.text = block.text.replace(/cch=[^;]+/g, "cch=<normalized>");
    }
  }

  if (Array.isArray(normalizedWithoutCacheControl.messages)) {
    for (const message of normalizedWithoutCacheControl.messages) {
      normalizeMessageContent(message);
    }
  }

  return normalizedWithoutCacheControl;
}

function resolvePromptCacheSessionId(req: any): string {
  if (typeof req?.sessionId === "string" && req.sessionId.length > 0) {
    return req.sessionId;
  }

  const metadataUserId = req?.body?.metadata?.user_id;
  if (typeof metadataUserId === "string" && metadataUserId.length > 0) {
    try {
      const parsed = JSON.parse(metadataUserId);
      if (typeof parsed?.session_id === "string" && parsed.session_id.length > 0) {
        return parsed.session_id;
      }
    } catch {}

    const sessionMatch = metadataUserId.match(/"session_id"\s*:\s*"([^"]+)"/);
    if (sessionMatch?.[1]) {
      return sessionMatch[1];
    }

    const legacyMatch = metadataUserId.match(/_session_([a-f0-9-]+)/i);
    if (legacyMatch?.[1]) {
      return legacyMatch[1];
    }
  }

  const headerValue = req?.headers?.session_id || req?.headers?.["session_id"];
  if (typeof headerValue === "string" && headerValue.length > 0) {
    return headerValue;
  }

  if (Array.isArray(headerValue) && typeof headerValue[0] === "string") {
    return headerValue[0];
  }

  return "";
}

function buildPromptCacheSessionPointerKey(
  sessionId: string,
  modelName: string,
  ttlType: PromptCacheTTL,
): string {
  return [
    "prompt-cache-session",
    modelName,
    ttlType,
    sessionId,
  ].join(":");
}

function arePromptCacheValuesEqual(left: any, right: any): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function isPromptCacheArrayPrefix(prefix: any[], target: any[]): boolean {
  if (prefix.length > target.length) {
    return false;
  }

  for (let index = 0; index < prefix.length; index += 1) {
    if (!arePromptCacheValuesEqual(prefix[index], target[index])) {
      return false;
    }
  }

  return true;
}

function isPromptCacheRequestPrefix(
  prefixRequest: Record<string, any>,
  targetRequest: Record<string, any>,
): boolean {
  const prefixSystem = prefixRequest.system;
  const targetSystem = targetRequest.system;

  if (prefixSystem !== undefined) {
    if (typeof prefixSystem === "string") {
      if (prefixSystem !== targetSystem) {
        return false;
      }
    } else if (Array.isArray(prefixSystem)) {
      if (!Array.isArray(targetSystem) || !isPromptCacheArrayPrefix(prefixSystem, targetSystem)) {
        return false;
      }
    } else if (!arePromptCacheValuesEqual(prefixSystem, targetSystem)) {
      return false;
    }
  }

  const prefixMessages = Array.isArray(prefixRequest.messages) ? prefixRequest.messages : [];
  const targetMessages = Array.isArray(targetRequest.messages) ? targetRequest.messages : [];
  return isPromptCacheArrayPrefix(prefixMessages, targetMessages);
}

function parsePromptCacheTTL(cacheControl: any): PromptCacheTTL {
  if (!cacheControl || typeof cacheControl !== "object") {
    return "5m";
  }

  const raw = [
    cacheControl.ttl,
    cacheControl.cache_ttl,
    cacheControl.duration,
    cacheControl.type,
    cacheControl.mode,
    cacheControl.name,
  ]
    .filter((item) => typeof item === "string" || typeof item === "number")
    .join(" ")
    .toLowerCase();

  if (
    raw.includes("1h") ||
    raw.includes("1 hour") ||
    raw.includes("1hour") ||
    raw.includes("60m") ||
    raw.includes("3600")
  ) {
    return "1h";
  }

  return "5m";
}

function hasCacheControl(value: any): boolean {
  return Boolean(value && typeof value === "object" && value.cache_control);
}

function buildPromptCacheCandidates(requestBody: any): PromptCacheCandidate[] {
  if (!requestBody || typeof requestBody !== "object") {
    return [];
  }

  const prefixRequest: Record<string, any> = {
    messages: [],
  };

  const snapshots: PromptCacheSnapshot[] = [];

  const captureSnapshot = (cacheControl: any) => {
    snapshots.push({
      request: cloneJSON(prefixRequest),
      ttlType: parsePromptCacheTTL(cacheControl),
    });
  };

  if (typeof requestBody.system === "string" && requestBody.system.length > 0) {
    prefixRequest.system = requestBody.system;
  } else if (Array.isArray(requestBody.system) && requestBody.system.length > 0) {
    prefixRequest.system = [];
    for (const block of requestBody.system) {
      prefixRequest.system.push(cloneJSON(block));
      if (hasCacheControl(block)) {
        captureSnapshot(block.cache_control);
      }
    }
  }

  for (const message of requestBody.messages || []) {
    const clonedMessage = cloneJSON({
      role: message.role,
      content: typeof message.content === "string" ? message.content : [],
    });
    prefixRequest.messages.push(clonedMessage);

    if (hasCacheControl(message)) {
      captureSnapshot(message.cache_control);
    }

    if (typeof message.content === "string") {
      continue;
    }

    if (!Array.isArray(message.content)) {
      clonedMessage.content = message.content;
      continue;
    }

    clonedMessage.content = [];
    for (const part of message.content) {
      clonedMessage.content.push(cloneJSON(part));
      if (hasCacheControl(part)) {
        captureSnapshot(part.cache_control);
      }
    }
  }

  if (snapshots.length === 0) {
    return [];
  }

  const modelName = String(requestBody.model || "");
  return snapshots.map((snapshot, index) => {
    const normalizedRequest = normalizePromptCacheRequest(snapshot.request);
    const normalized = JSON.stringify(normalizedRequest);
    const prefixHash = createHash("sha256").update(normalized).digest("hex");
    const cacheKey = [
      "prompt-cache",
      modelName,
      snapshot.ttlType,
      prefixHash,
    ].join(":");

    return {
      cacheKey,
      normalizedPrefixRequest: normalizedRequest,
      prefixHash,
      prefixRequest: snapshot.request,
      ttlType: snapshot.ttlType,
      snapshotIndex: index,
    };
  });
}

async function getPromptCacheEntry(cacheKey: string): Promise<PromptCacheEntry | null> {
  if (!promptCacheStore) {
    throw new Error("prompt cache redis store is not initialized");
  }

  return promptCacheStore.get(cacheKey);
}

async function getPromptCacheSessionPointer(
  sessionKey: string,
): Promise<PromptCacheSessionPointer | null> {
  if (!promptCacheStore) {
    throw new Error("prompt cache redis store is not initialized");
  }

  const parsed = await promptCacheStore.getJSON(sessionKey);
  if (!parsed || typeof parsed !== "object") {
    return null;
  }

  if (Number(parsed.expiresAt || 0) <= Date.now()) {
    await promptCacheStore.delete(sessionKey);
    return null;
  }

  return {
    expiresAt: Math.max(0, Math.round(Number(parsed.expiresAt || 0))),
    key: String(parsed.key || sessionKey),
    normalizedPrefixRequest: parsed.normalizedPrefixRequest && typeof parsed.normalizedPrefixRequest === "object"
      ? parsed.normalizedPrefixRequest
      : { messages: [] },
    prefixHash: String(parsed.prefixHash || ""),
    prefixTokens: Math.max(0, Math.round(Number(parsed.prefixTokens || 0))),
    ttlType: parsed.ttlType === "1h" ? "1h" : "5m",
    updatedAt: Math.max(0, Math.round(Number(parsed.updatedAt || 0))),
  };
}

async function storePromptCacheEntry(descriptor: PromptCacheDescriptor, prefixTokens: number): Promise<PromptCacheEntry> {
  if (!promptCacheStore) {
    throw new Error("prompt cache redis store is not initialized");
  }

  const now = Date.now();
  const entry: PromptCacheEntry = {
    key: descriptor.cacheKey,
    ttlType: descriptor.ttlType,
    prefixTokens,
    createdAt: now,
    expiresAt: now + PROMPT_CACHE_TTL_MS[descriptor.ttlType],
  };
  await promptCacheStore.setIfAbsent(entry);
  return entry;
}

async function storePromptCacheSessionPointer(pointer: PromptCacheSessionPointer): Promise<void> {
  if (!promptCacheStore) {
    throw new Error("prompt cache redis store is not initialized");
  }

  await promptCacheStore.setJSON(
    pointer.key,
    pointer,
    Math.max(1, pointer.expiresAt - Date.now()),
  );
}

function buildPromptCacheUsage(inputTokens: number, cacheUsage: PromptCacheUsage) {
  const usage: Record<string, any> = {
    input_tokens: Math.max(0, Math.round(inputTokens)),
    output_tokens: 0,
    cache_creation_input_tokens: Math.max(0, Math.round(cacheUsage.cacheCreationInputTokens)),
    cache_read_input_tokens: Math.max(0, Math.round(cacheUsage.cacheReadInputTokens)),
  };

  if (cacheUsage.cacheCreation5mTokens > 0 || cacheUsage.cacheCreation1hTokens > 0) {
    usage.cache_creation = {
      ephemeral_5m_input_tokens: Math.max(0, Math.round(cacheUsage.cacheCreation5mTokens)),
      ephemeral_1h_input_tokens: Math.max(0, Math.round(cacheUsage.cacheCreation1hTokens)),
    };
  }

  return usage;
}

async function resolvePromptCacheUsage(
  serverInstance: any,
  req: any,
  totalInputTokens: number,
): Promise<PromptCacheUsage> {
  const candidates = buildPromptCacheCandidates(req.body);
  if (candidates.length === 0) {
    return {
      inputTokens: Math.max(0, totalInputTokens),
      cacheCreationInputTokens: 0,
      cacheReadInputTokens: 0,
      cacheCreation5mTokens: 0,
      cacheCreation1hTokens: 0,
      shouldStore: false,
      stored: false,
    };
  }

  const tokenizedCandidates = await Promise.all(
    candidates.map(async (candidate) => {
      const prefixTokens = Math.min(
        totalInputTokens,
        await countAnthropicTokens(serverInstance, req, {
          messages: candidate.prefixRequest.messages || [],
          system: candidate.prefixRequest.system,
        }),
      );
      return {
        candidate,
        prefixTokens,
        existing: await getPromptCacheEntry(candidate.cacheKey),
      };
    }),
  );

  const finalCandidate = tokenizedCandidates[tokenizedCandidates.length - 1];
  const descriptor = finalCandidate.candidate;
  const prefixTokens = finalCandidate.prefixTokens;
  const sessionId = resolvePromptCacheSessionId(req);
  const sessionPointerKey = sessionId
    ? buildPromptCacheSessionPointerKey(sessionId, String(req.body?.model || ""), descriptor.ttlType)
    : "";
  const sessionPointer = sessionPointerKey
    ? await getPromptCacheSessionPointer(sessionPointerKey)
    : null;

  if (prefixTokens <= 0) {
    return {
      inputTokens: Math.max(0, totalInputTokens),
      cacheCreationInputTokens: 0,
      cacheReadInputTokens: 0,
      cacheCreation5mTokens: 0,
      cacheCreation1hTokens: 0,
      shouldStore: false,
      stored: false,
      sessionPointer: undefined,
    };
  }

  const bestReusableCandidate = tokenizedCandidates
    .filter((item) => item.existing && item.prefixTokens > 0)
    .sort((left, right) => right.prefixTokens - left.prefixTokens)[0];
  const explicitReusablePrefixTokens = bestReusableCandidate?.prefixTokens || 0;
  const sessionReusablePrefixTokens = sessionPointer &&
    isPromptCacheRequestPrefix(
      sessionPointer.normalizedPrefixRequest,
      descriptor.normalizedPrefixRequest,
    )
    ? Math.min(prefixTokens, sessionPointer.prefixTokens)
    : 0;
  const reusablePrefixTokens = Math.max(
    explicitReusablePrefixTokens,
    sessionReusablePrefixTokens,
  );
  const nonCachedInputTokens = Math.max(0, totalInputTokens - prefixTokens);
  const cacheCreationInputTokens = Math.max(0, prefixTokens - reusablePrefixTokens);
  const storeCandidates = tokenizedCandidates
    .filter((item) => !item.existing && item.prefixTokens > reusablePrefixTokens)
    .map((item) => ({
      descriptor: item.candidate,
      prefixTokens: item.prefixTokens,
    }));

  if (bestReusableCandidate && reusablePrefixTokens >= prefixTokens) {
    return {
      inputTokens: nonCachedInputTokens,
      cacheCreationInputTokens: 0,
      cacheReadInputTokens: prefixTokens,
      cacheCreation5mTokens: 0,
      cacheCreation1hTokens: 0,
      shouldStore: false,
      stored: true,
      descriptor,
      sessionPointer: sessionPointerKey
        ? {
            key: sessionPointerKey,
            ttlType: descriptor.ttlType,
            prefixHash: descriptor.prefixHash,
            prefixTokens,
            normalizedPrefixRequest: descriptor.normalizedPrefixRequest,
            updatedAt: Date.now(),
            expiresAt: Date.now() + PROMPT_CACHE_TTL_MS[descriptor.ttlType],
          }
        : undefined,
      storeCandidates: [],
    };
  }

  return {
    inputTokens: nonCachedInputTokens,
    cacheCreationInputTokens,
    cacheReadInputTokens: reusablePrefixTokens,
    cacheCreation5mTokens: descriptor.ttlType === "5m" ? cacheCreationInputTokens : 0,
    cacheCreation1hTokens: descriptor.ttlType === "1h" ? cacheCreationInputTokens : 0,
    shouldStore: storeCandidates.length > 0,
    stored: false,
    descriptor,
    sessionPointer: sessionPointerKey
      ? {
          key: sessionPointerKey,
          ttlType: descriptor.ttlType,
          prefixHash: descriptor.prefixHash,
          prefixTokens,
          normalizedPrefixRequest: descriptor.normalizedPrefixRequest,
          updatedAt: Date.now(),
          expiresAt: Date.now() + PROMPT_CACHE_TTL_MS[descriptor.ttlType],
        }
      : undefined,
    storeCandidates,
  };
}

async function ensurePromptCacheStored(cacheUsage: PromptCacheUsage) {
  if (!cacheUsage.shouldStore || cacheUsage.stored || !cacheUsage.storeCandidates?.length) {
    return;
  }

  for (const candidate of cacheUsage.storeCandidates) {
    await storePromptCacheEntry(candidate.descriptor, candidate.prefixTokens);
  }
  cacheUsage.stored = true;
}

async function ensurePromptCacheSessionPointerStored(cacheUsage: PromptCacheUsage) {
  if (!cacheUsage.sessionPointer) {
    return;
  }

  await storePromptCacheSessionPointer(cacheUsage.sessionPointer);
}

function hasAnthropicUsage(usage: any): boolean {
  if (!usage || typeof usage !== "object") {
    return false;
  }

  return (
    Number(usage.input_tokens || 0) > 0 ||
    Number(usage.output_tokens || 0) > 0 ||
    Number(usage.cache_creation_input_tokens || 0) > 0 ||
    Number(usage.cache_read_input_tokens || 0) > 0
  );
}

function buildAnthropicUsage(inputTokens: number, outputTokens: number, cacheUsage?: PromptCacheUsage) {
  const usage = buildPromptCacheUsage(inputTokens, cacheUsage || {
    inputTokens,
    cacheCreationInputTokens: 0,
    cacheReadInputTokens: 0,
    cacheCreation5mTokens: 0,
    cacheCreation1hTokens: 0,
    shouldStore: false,
    stored: false,
  });
  usage.output_tokens = Math.max(0, Math.round(outputTokens));
  return usage;
}

function extractAnthropicStreamText(eventData: any): string {
  if (eventData?.event === "content_block_start") {
    if (eventData?.data?.content_block?.type === "tool_use") {
      return eventData.data.content_block.name || "";
    }
    return "";
  }

  if (eventData?.event !== "content_block_delta") {
    return "";
  }

  const delta = eventData?.data?.delta;
  if (!delta) {
    return "";
  }

  if (delta.type === "text_delta") {
    return delta.text || "";
  }

  if (delta.type === "thinking_delta") {
    return delta.thinking || "";
  }

  if (delta.type === "input_json_delta") {
    return delta.partial_json || "";
  }

  return "";
}

async function countAnthropicTokens(
  serverInstance: any,
  req: any,
  tokenRequest: any,
): Promise<number> {
  const tokenizerService = serverInstance?.app?._server?.tokenizerService;
  const providerName = req.provider;
  const modelName = req.body?.model;

  if (tokenizerService && providerName && modelName) {
    try {
      const tokenizerConfig = tokenizerService.getTokenizerConfigForModel(
        providerName,
        modelName,
      );
      const result = await tokenizerService.countTokens(
        tokenRequest,
        tokenizerConfig,
      );
      return Math.max(0, Math.round(result?.tokenCount || 0));
    } catch (error) {
      req.log?.warn({ error }, "failed to count tokens with tokenizer service");
    }
  }

  return Math.max(
    0,
    Math.round(
      calculateTokenCount(
        tokenRequest.messages || [],
        tokenRequest.system,
        tokenRequest.tools,
      ),
    ),
  );
}

/**
 * Register plugins from configuration
 * @param serverInstance Server instance
 * @param config Application configuration
 */
async function registerPluginsFromConfig(serverInstance: any, config: any): Promise<void> {
  // Get plugins configuration from config file
  const pluginsConfig: PluginConfig[] = config.plugins || config.Plugins || [];

  for (const pluginConfig of pluginsConfig) {
      const { name, enabled = false, options = {} } = pluginConfig;

      switch (name) {
        case 'token-speed':
          pluginManager.registerPlugin(tokenSpeedPlugin, {
            enabled,
            outputHandlers: [
              {
                type: 'temp-file',
                enabled: true
              }
            ],
            ...options
          });
          break;

        default:
          console.warn(`Unknown plugin: ${name}`);
          break;
      }
    }
  // Enable all registered plugins
  await pluginManager.enablePlugins(serverInstance);
}

async function getServer(options: RunOptions = {}) {
  await initializeClaudeConfig();
  await initDir();
  const config = await initConfig();

  // Check if Providers is configured
  const providers = config.Providers || config.providers || [];
  const hasProviders = providers && providers.length > 0;

  let HOST = config.HOST || "127.0.0.1";

  if (hasProviders) {
    HOST = config.HOST;
    if (!config.APIKEY) {
      HOST = "127.0.0.1";
    }
  } else {
    // When no providers are configured, listen on 0.0.0.0 without authentication
    HOST = "0.0.0.0";
    console.log("ℹ️  No providers configured. Listening on 0.0.0.0 without authentication.");
  }

  const port = config.PORT || 3456;

  // Use port from environment variable if set (for background process)
  const servicePort = process.env.SERVICE_PORT
    ? parseInt(process.env.SERVICE_PORT)
    : port;
  const promptCacheRedisUrl = process.env.PROMPT_CACHE_REDIS_URL;

  if (!promptCacheRedisUrl) {
    throw new Error("PROMPT_CACHE_REDIS_URL is required");
  }

  promptCacheStore = new PromptCacheRedisStore(promptCacheRedisUrl);
  await promptCacheStore.ping();

  // Configure logger based on config settings or external options
  const pad = (num: number) => (num > 9 ? "" : "0") + num;
  const generator = (time: number | Date | undefined, index: number | undefined) => {
    let date: Date;
    if (!time) {
      date = new Date();
    } else if (typeof time === 'number') {
      date = new Date(time);
    } else {
      date = time;
    }

    const month = date.getFullYear() + "" + pad(date.getMonth() + 1);
    const day = pad(date.getDate());
    const hour = pad(date.getHours());
    const minute = pad(date.getMinutes());

    return `./logs/ccr-${month}${day}${hour}${minute}${pad(date.getSeconds())}${index ? `_${index}` : ''}.log`;
  };

  let loggerConfig: any;

  // Use external logger configuration if provided
  if (options.logger !== undefined) {
    loggerConfig = options.logger;
  } else {
    // Enable logger if not provided and config.LOG !== false
    if (config.LOG !== false) {
      // Set config.LOG to true (if not already set)
      if (config.LOG === undefined) {
        config.LOG = true;
      }
      loggerConfig = {
        level: config.LOG_LEVEL || "debug",
        stream: createStream(generator, {
          path: HOME_DIR,
          maxFiles: 3,
          interval: "1d",
          compress: false,
          maxSize: "50M"
        }),
      };
    } else {
      loggerConfig = false;
    }
  }

  const presets = await listPresets();

  const serverInstance = await createServer({
    jsonPath: CONFIG_FILE,
    initialConfig: {
      // ...config,
      providers: config.Providers || config.providers,
      HOST: HOST,
      PORT: servicePort,
      LOG_FILE: join(
        homedir(),
        ".claude-code-router",
        "claude-code-router.log"
      ),
    },
    logger: loggerConfig,
  });

  await Promise.allSettled(
      presets.map(async preset => await serverInstance.registerNamespace(`/preset/${preset.name}`, preset.config))
  )

  // Register and configure plugins from config
  await registerPluginsFromConfig(serverInstance, config);

  // Add async preHandler hook for authentication
  serverInstance.addHook("preHandler", async (req: any, reply: any) => {
    return new Promise<void>((resolve, reject) => {
      const done = (err?: Error) => {
        if (err) reject(err);
        else resolve();
      };
      // Call the async auth function
      apiKeyAuth(config)(req, reply, done).catch(reject);
    });
  });
  serverInstance.addHook("preHandler", async (req: any, reply: any) => {
    const url = new URL(`http://127.0.0.1${req.url}`);
    req.pathname = url.pathname;
    if (req.pathname.endsWith("/v1/messages") && req.pathname !== "/v1/messages") {
      req.preset = req.pathname.replace("/v1/messages", "").replace("/", "");
    }
  })

  serverInstance.addHook("preHandler", async (req: any, reply: any) => {
    if (req.pathname.endsWith("/v1/messages")) {
      const useAgents = []

      for (const agent of agentsManager.getAllAgents()) {
        if (agent.shouldHandle(req, config)) {
          // Set agent identifier
          useAgents.push(agent.name)

          // change request body
          agent.reqHandler(req, config);

          // append agent tools
          if (agent.tools.size) {
            if (!req.body?.tools?.length) {
              req.body.tools = []
            }
            req.body.tools.unshift(...Array.from(agent.tools.values()).map(item => {
              return {
                name: item.name,
                description: item.description,
                input_schema: item.input_schema
              }
            }))
          }
        }
      }

      if (useAgents.length) {
        req.agents = useAgents;
      }
    }
  });
  serverInstance.addHook("onError", async (request: any, reply: any, error: any) => {
    event.emit('onError', request, reply, error);
  })
  serverInstance.addHook("onSend", (req: any, reply: any, payload: any, done: any) => {
    if (req.sessionId && req.pathname.endsWith("/v1/messages")) {
      if (payload instanceof ReadableStream) {
        if (req.agents) {
          const abortController = new AbortController();
          const eventStream = payload.pipeThrough(new SSEParserTransform())
          let currentAgent: undefined | IAgent;
          let currentToolIndex = -1
          let currentToolName = ''
          let currentToolArgs = ''
          let currentToolId = ''
          const toolMessages: any[] = []
          const assistantMessages: any[] = []
          // Store Anthropic format message body, distinguishing text and tool types
          return done(null, rewriteStream(eventStream, async (data, controller) => {
            try {
              // Detect tool call start
              if (data.event === 'content_block_start' && data?.data?.content_block?.name) {
                const agent = req.agents.find((name: string) => agentsManager.getAgent(name)?.tools.get(data.data.content_block.name))
                if (agent) {
                  currentAgent = agentsManager.getAgent(agent)
                  currentToolIndex = data.data.index
                  currentToolName = data.data.content_block.name
                  currentToolId = data.data.content_block.id
                  return undefined;
                }
              }

              // Collect tool arguments
              if (currentToolIndex > -1 && data.data.index === currentToolIndex && data.data?.delta?.type === 'input_json_delta') {
                currentToolArgs += data.data?.delta?.partial_json;
                return undefined;
              }

              // Tool call completed, handle agent invocation
              if (currentToolIndex > -1 && data.data.index === currentToolIndex && data.data.type === 'content_block_stop') {
                try {
                  const args = JSON5.parse(currentToolArgs);
                  assistantMessages.push({
                    type: "tool_use",
                    id: currentToolId,
                    name: currentToolName,
                    input: args
                  })
                  const toolResult = await currentAgent?.tools.get(currentToolName)?.handler(args, {
                    req,
                    config
                  });
                  toolMessages.push({
                    "tool_use_id": currentToolId,
                    "type": "tool_result",
                    "content": toolResult
                  })
                  currentAgent = undefined
                  currentToolIndex = -1
                  currentToolName = ''
                  currentToolArgs = ''
                  currentToolId = ''
                } catch (e) {
                  console.log(e);
                }
                return undefined;
              }

              if (data.event === 'message_delta' && toolMessages.length) {
                req.body.messages.push({
                  role: 'assistant',
                  content: assistantMessages
                })
                req.body.messages.push({
                  role: 'user',
                  content: toolMessages
                })
                const response = await fetch(`http://127.0.0.1:${config.PORT || 3456}/v1/messages`, {
                  method: "POST",
                  headers: {
                    'x-api-key': config.APIKEY,
                    'content-type': 'application/json',
                  },
                  body: JSON.stringify(req.body),
                })
                if (!response.ok) {
                  return undefined;
                }
                const stream = response.body!.pipeThrough(new SSEParserTransform() as any)
                const reader = stream.getReader()
                while (true) {
                  try {
                    const {value, done} = await reader.read();
                    if (done) {
                      break;
                    }
                    const eventData = value as any;
                    if (['message_start', 'message_stop'].includes(eventData.event)) {
                      continue
                    }

                    // Check if stream is still writable
                    if (!controller.desiredSize) {
                      break;
                    }

                    controller.enqueue(eventData)
                  }catch (readError: any) {
                    if (readError.name === 'AbortError' || readError.code === 'ERR_STREAM_PREMATURE_CLOSE') {
                      abortController.abort(); // Abort all related operations
                      break;
                    }
                    throw readError;
                  }

                }
                return undefined
              }
              return data
            }catch (error: any) {
              console.error('Unexpected error in stream processing:', error);

              // Handle premature stream closure error
              if (error.code === 'ERR_STREAM_PREMATURE_CLOSE') {
                abortController.abort();
                return undefined;
              }

              // Re-throw other errors
              throw error;
            }
          }).pipeThrough(new SSESerializerTransform()))
        }

        const [originalStream, clonedStream] = payload.tee();
        const read = async (stream: ReadableStream) => {
          const reader = stream.getReader();
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              // Process the value if needed
              const dataStr = new TextDecoder().decode(value);
              if (!dataStr.startsWith("event: message_delta")) {
                continue;
              }
              const str = dataStr.slice(27);
              try {
                const message = JSON.parse(str);
                sessionUsageCache.put(req.sessionId, message.usage);
              } catch {}
            }
          } catch (readError: any) {
            if (readError.name === 'AbortError' || readError.code === 'ERR_STREAM_PREMATURE_CLOSE') {
              console.error('Background read stream closed prematurely');
            } else {
              console.error('Error in background stream reading:', readError);
            }
          } finally {
            reader.releaseLock();
          }
        }
        read(clonedStream);
        return done(null, originalStream)
      }
      sessionUsageCache.put(req.sessionId, payload.usage);
      if (typeof payload ==='object') {
        if (payload.error) {
          return done(payload.error, null)
        } else {
          return done(payload, null)
        }
      }
    }
    if (typeof payload ==='object' && payload.error) {
      return done(payload.error, null)
    }
    done(null, payload)
  });
  serverInstance.addHook("onSend", async (req: any, reply: any, payload: any) => {
    if (!req.pathname?.endsWith("/v1/messages")) {
      return payload;
    }

    const totalInputTokensPromise = countAnthropicTokens(serverInstance, req, {
      messages: req.body?.messages || [],
      system: req.body?.system,
      tools: req.body?.tools,
    });
    const promptCacheUsagePromise = totalInputTokensPromise.then((totalInputTokens) =>
      resolvePromptCacheUsage(serverInstance, req, totalInputTokens),
    );

    if (!(payload instanceof ReadableStream)) {
      if (payload && typeof payload === "object" && !payload.error) {
        const promptCacheUsage = await promptCacheUsagePromise;
        if (!hasAnthropicUsage(payload.usage)) {
          await ensurePromptCacheStored(promptCacheUsage);
          await ensurePromptCacheSessionPointerStored(promptCacheUsage);
          payload.usage = buildAnthropicUsage(
            promptCacheUsage.inputTokens,
            Number(payload.usage?.output_tokens || 0),
            promptCacheUsage,
          );
        }
      }
      return payload;
    }

    let completionText = "";
    let usagePatched = false;

    return rewriteStream(
      payload
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new SSEParserTransform() as any),
      async (eventData, controller) => {
        completionText += extractAnthropicStreamText(eventData);

        if (eventData?.event === "message_start") {
          const promptCacheUsage = await promptCacheUsagePromise;
          await ensurePromptCacheStored(promptCacheUsage);
          await ensurePromptCacheSessionPointerStored(promptCacheUsage);
          const originalUsage = eventData?.data?.message?.usage || {};
          eventData.data.message.usage = {
            ...buildAnthropicUsage(promptCacheUsage.inputTokens, 0, promptCacheUsage),
            output_tokens: Number(originalUsage.output_tokens || 0),
          };
          return eventData;
        }

        if (eventData?.event === "message_delta") {
          if (!hasAnthropicUsage(eventData?.data?.usage)) {
            const promptCacheUsage = await promptCacheUsagePromise;
            const outputTokens = await countAnthropicTokens(serverInstance, req, {
              messages: [
                {
                  role: "assistant",
                  content: completionText,
                },
              ],
            });

            await ensurePromptCacheStored(promptCacheUsage);
            await ensurePromptCacheSessionPointerStored(promptCacheUsage);
            usagePatched = true;
            return {
              ...eventData,
              data: {
                ...eventData.data,
                usage: buildAnthropicUsage(
                  promptCacheUsage.inputTokens,
                  outputTokens,
                  promptCacheUsage,
                ),
              },
            };
          }

          usagePatched = true;
          return eventData;
        }

        if (eventData?.event === "message_stop" && !usagePatched) {
          const promptCacheUsage = await promptCacheUsagePromise;
          const outputTokens = await countAnthropicTokens(serverInstance, req, {
            messages: [
              {
                role: "assistant",
                content: completionText,
              },
            ],
          });

          await ensurePromptCacheStored(promptCacheUsage);
          await ensurePromptCacheSessionPointerStored(promptCacheUsage);
          (controller as any).enqueue({
            event: "message_delta",
            data: {
              type: "message_delta",
              delta: {
                stop_reason: "end_turn",
                stop_sequence: null,
              },
              usage: buildAnthropicUsage(
                promptCacheUsage.inputTokens,
                outputTokens,
                promptCacheUsage,
              ),
            },
          });
          usagePatched = true;
          return eventData;
        }

        return eventData;
      },
    )
      .pipeThrough(new SSESerializerTransform() as any)
      .pipeThrough(new TextEncoderStream());
  });
  serverInstance.addHook("onSend", async (req: any, reply: any, payload: any) => {
    event.emit('onSend', req, reply, payload);
    return payload;
  });

  // Add global error handlers to prevent the service from crashing
  process.on("uncaughtException", (err) => {
    serverInstance.app.log.error("Uncaught exception:", err);
  });

  process.on("unhandledRejection", (reason, promise) => {
    serverInstance.app.log.error("Unhandled rejection at:", promise, "reason:", reason);
  });

  return serverInstance;
}

async function run() {
  const server = await getServer();
  server.app.post("/api/restart", async () => {
    setTimeout(async () => {
      process.exit(0);
    }, 100);

    return { success: true, message: "Service restart initiated" }
  });
  await server.start();
}

export { getServer };
export type { RunOptions };
export type { IAgent, ITool } from "./agents/type";
export { initDir, initConfig, readConfigFile, writeConfigFile, backupConfigFile } from "./utils";
export { pluginManager, tokenSpeedPlugin } from "@musistudio/llms";

// Start service if this file is run directly
if (require.main === module) {
  run().catch((error) => {
    console.error('Failed to start server:', error);
    process.exit(1);
  });
}
