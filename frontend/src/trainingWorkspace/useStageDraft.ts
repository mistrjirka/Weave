import { useEffect, useState } from "react";
import { z } from "zod";
import type { TrainingStage } from "../trainingApi";
import { previewSchema } from "./StageData";
export function useStageDraft(stage: TrainingStage | undefined) {
  const config = stage?.config;
  const [preview, setPreview] = useState<z.infer<typeof previewSchema> | null>(
    null,
  );
  const [promptsText, setPromptsText] = useState("");
  useEffect(() => {
    setPreview(null);
  }, [
    stage?.id,
    config?.dataset,
    config?.sequence_length,
    config?.recipe,
    config?.chat_template,
  ]);
  useEffect(() => {
    setPromptsText(JSON.stringify(stage?.config.prompts ?? [], null, 2));
  }, [stage?.id]);
  return { preview, setPreview, promptsText, setPromptsText };
}
