import { useState, type Dispatch, type SetStateAction } from "react";
import { ShieldCheck } from "lucide-react";
import { z } from "zod";
import { capabilitySchema, trainingRequest } from "../trainingApi";
const responseOK = z.object({ authenticated: z.boolean() });
const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);
type Capabilities = z.infer<typeof capabilitySchema>;
type Props = {
  capabilities: Capabilities | null;
  loaded: boolean;
  error: string;
  monitorError: string;
  busy: boolean;
  task: (action: () => Promise<unknown>) => Promise<void>;
  setError: (error: string) => void;
  setCapabilities: (capabilities: Capabilities) => void;
  setBootstrapRetry: Dispatch<SetStateAction<number>>;
};
export function TrainingAccess({
  capabilities,
  loaded,
  error,
  monitorError,
  busy,
  task,
  setError,
  setCapabilities,
  setBootstrapRetry,
}: Props) {
  const [token, setToken] = useState("");
  if (!capabilities || (capabilities.authenticated && !loaded))
    return (
      <section
        className="training-workspace training-loading"
        aria-label="Training workspace"
        aria-busy={!error}
      >
        {!error && <p role="status">Loading training workspace…</p>}
        {error || monitorError ? (
          <div role="alert">
            {error || monitorError}{" "}
            <button
              onClick={() => {
                setError("");
                if (!capabilities)
                  void trainingRequest("/capabilities", capabilitySchema)
                    .then(setCapabilities)
                    .catch((e) => setError(message(e)));
                else setBootstrapRetry((value) => value + 1);
              }}
            >
              Retry
            </button>
          </div>
        ) : (
          <div className="training-loading-layout" aria-hidden="true">
            <div className="training-loading-bar" />
            <div className="training-loading-card">
              <div className="training-loading-bar" />
              <div className="training-loading-plots">
                <div />
                <div />
                <div />
                <div />
              </div>
            </div>
          </div>
        )}
      </section>
    );
  if (!capabilities.authenticated)
    return (
      <section className="training-workspace training-unlock">
        <ShieldCheck aria-hidden="true" size={32} />
        <h2>Train and track your model</h2>
        <p>
          Create editable training stages, watch generated examples, and keep
          checkpoints with their data and scripts.
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void task(async () => {
              await trainingRequest("/session", responseOK, { token });
              setToken("");
              setCapabilities(
                await trainingRequest("/capabilities", capabilitySchema),
              );
            });
          }}
        >
          <label>
            Operator access code
            <input
              type="password"
              autoComplete="current-password"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              required
            />
          </label>
          <button className="primary" disabled={busy}>
            Unlock training
          </button>
        </form>
        <p className="training-muted">
          This server has optional training authentication enabled. Use the
          WEAVE_TRAINING_TOKEN supplied by its operator.
        </p>
        {!capabilities.enabled && (
          <p>
            Local launch is disabled. Run{" "}
            <code>bash scripts/install.sh --training cpu</code> on the server,
            or configure an existing training Python, then restart Weave.
          </p>
        )}
        {error && <p role="alert">{error}</p>}
      </section>
    );

  return null;
}
