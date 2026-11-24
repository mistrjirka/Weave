import {
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import { z } from "zod";
import { trainingRequest } from "../trainingApi";
import { sessionSchema } from "./chatSchema";

export function useChatSession(
  endpoint: string,
  setPrompt: Dispatch<SetStateAction<string>>,
  setToolsNotice: Dispatch<SetStateAction<string>>,
) {
  const [session, setSession] = useState<z.infer<typeof sessionSchema> | null>(
    null,
  );
  const [error, setError] = useState("");
  const [pollError, setPollError] = useState("");
  const loadTimedOut = useRef(false);
  const [busy, setBusy] = useState(false);
  const requestPending = useRef(false);
  const operation = useRef(0);
  useEffect(() => {
    let disposed = false;
    let pending = false;
    let current: AbortController | null = null;
    const refresh = async () => {
      if (pending || requestPending.current) return;
      pending = true;
      const revision = operation.current;
      const controller = new AbortController();
      current = controller;
      const timeout = window.setTimeout(() => controller.abort(), 15000);
      try {
        const result = await trainingRequest(
          endpoint + "/session",
          sessionSchema,
          undefined,
          undefined,
          controller.signal,
        );
        if (!disposed && revision === operation.current) {
          setSession(result);
          setPollError("");
          if (loadTimedOut.current && result.state === "ready") {
            loadTimedOut.current = false;
            setError("");
          }
        }
      } catch (failure) {
        if (!disposed)
          setPollError(
            controller.signal.aborted
              ? "Model status request timed out. Retrying automatically."
              : `Cannot refresh model status: ${failure instanceof Error ? failure.message : String(failure)}. Retrying automatically.`,
          );
      } finally {
        window.clearTimeout(timeout);
        current = null;
        pending = false;
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1000);
    return () => {
      disposed = true;
      current?.abort();
      window.clearInterval(timer);
    };
  }, [endpoint]);

  async function action(kind: string, body: unknown = {}, method = "POST") {
    operation.current += 1;
    requestPending.current = true;
    setBusy(true);
    setError("");
    loadTimedOut.current = false;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    try {
      const result = await trainingRequest(
        endpoint + kind,
        sessionSchema,
        body,
        method,
        controller.signal,
      );
      setSession(result);
      if (kind === "/messages") setPrompt("");
      if (kind === "/settings")
        setToolsNotice("Settings saved for this project.");
      return result;
    } catch (failure) {
      loadTimedOut.current = kind === "/load" && controller.signal.aborted;
      setError(
        controller.signal.aborted
          ? "The request took too long. Model status checks will continue; you can unload and retry."
          : failure instanceof Error
            ? failure.message
            : String(failure),
      );
      return null;
    } finally {
      window.clearTimeout(timeout);
      requestPending.current = false;
      setBusy(false);
    }
  }
  return { session, setSession, error, setError, pollError, busy, action };
}
