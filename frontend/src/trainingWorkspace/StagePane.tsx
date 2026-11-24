import type { ReactNode } from "react";
import { StageActiveNotice } from "./StageLaunchControls";

export function StagePane({
  active,
  children,
}: {
  active: boolean;
  children: ReactNode;
}) {
  return (
    <section
      className="training-card training-setup-card"
      aria-label="Stage setup"
    >
      <StageActiveNotice active={active} />
      <div className="training-stage-body">{children}</div>
    </section>
  );
}
