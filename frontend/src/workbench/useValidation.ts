import { useEffect, useMemo, useState } from "react";
import { codeSchema, describeError, responseData } from "../api";
import { compilationInput } from "../compilationInput";
import { connectionFeedback, type ConnectionAttempt } from "../connectionFeedback";
import type { Project, ValidationReport } from "../types";

export function useValidation(project: Project, connectionAttempt: ConnectionAttempt | null) {
  const [validatedInput, setValidatedInput] = useState("");
  const [lastReport, setReport] = useState<ValidationReport | null>(null);
  const [lastSource, setSource] = useState("");
  const [validationFailure, setValidationFailure] = useState<{ input: string; message: string } | null>(null);
  const compilationBody = useMemo(() => compilationInput(project), [project]);
  const report = validatedInput === compilationBody ? lastReport : null;
  const source = validatedInput === compilationBody ? lastSource : "";
  const validationError = validationFailure?.input === compilationBody ? validationFailure.message : "";
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    const timer = setTimeout(() => {
      fetch("/api/code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: compilationBody,
        signal: controller.signal,
      })
        .then(responseData)
        .then((data) => {
          const parsed = codeSchema.parse(data);
          if (active && parsed.report) {
            setReport(parsed.report);
            setValidationFailure(null);
            setValidatedInput(compilationBody);
            setSource(parsed.source);
          }
        })
        .catch((error) => {
          if (active && error.name !== "AbortError")
            setValidationFailure({ input: compilationBody, message: describeError(error) });
        });
    }, 220);
    return () => {
      active = false;
      clearTimeout(timer);
      controller.abort();
    };
  }, [compilationBody]);
  const connectionError = connectionAttempt && report && validatedInput === compilationBody
    ? connectionFeedback(project, report, connectionAttempt)
    : null;
  const clearValidation = () => { setReport(null); setSource(""); };
  return { report, source, validationError, connectionError, clearValidation };
}
