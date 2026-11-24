import React, { useEffect } from "react";
import { Handle, Position, useUpdateNodeInternals, type NodeProps } from "@xyflow/react";
import { ArrowUpRight, CircleAlert, Database, Shapes } from "lucide-react";
import { multiOutputLanes, portTop, portLabel } from "../lanePorts";
import { tokenizerLabel } from "../TokenizerPanel";
import { inferRole } from "../graph";
import type { DisplayNode } from "../types";
import { Diagram } from "./Diagram";
import { icons, family } from "./presentation";
import { format, motion } from "./util";
import { useReactFlow, useViewport } from "@xyflow/react";

function ArchitectureNode({ id, data, selected }: NodeProps<DisplayNode>) {
  const kind = data.kind;
  const Icon = icons[kind] || Shapes;
  const p = data._previewParams ?? data.params;
  const expanded = ["loop", "block", "moe"].includes(kind);
  const role = data._role || inferRole(data);
  const color = role === "denoising" ? "coral" : family(kind);
  const shape = data._shape;
  const updateInternals = useUpdateNodeInternals();
  const portSignature = JSON.stringify([data._ports, data._outputs]);
  useEffect(() => {
    updateInternals(id);
  }, [id, portSignature, updateInternals]);
  const laneNode = multiOutputLanes.has(kind) || kind === "lane_join";
  return (
    <div
      className={`arch-node ${color} role-${role} ${expanded ? "container-node" : ""} ${selected || data._focused ? "selected" : ""} ${data._error ? "has-error" : ""} ${laneNode ? "explicit-lane-node" : ""}`}
      style={
        kind === "moe_combine" || laneNode
          ? {
              minHeight:
                105 + Math.max(data._ports.length, data._outputs.length) * 28,
            }
          : undefined
      }
      data-visual-role={role}
      aria-label={`${data.label} node`}
    >
      {data._ports.map((port: string, i: number) => (
        <React.Fragment key={port}>
          <Handle
            type="target"
            position={Position.Left}
            id={port}
            title={
              data._portErrors?.[port]
                ? `Input ${port}: ${data._portErrors[port]}`
                : data._optional.includes(port)
                  ? `${port}: optional attention pattern / invariant`
                  : `Input ${port}`
            }
            className={`${data._optional.includes(port) ? "optional-port" : ""} ${data._portErrors?.[port] ? "port-error" : ""}`}
            style={{ top: portTop(kind) + i * 28 }}
          />
          <span
            className={`port-name ${data._portErrors?.[port] ? "port-error-label" : ""}`}
            style={{ top: portTop(kind) - 8 + i * 28 }}
          >
            {portLabel(port)}
          </span>
        </React.Fragment>
      ))}
      {data._outputs.map((port, i) => (
        <React.Fragment key={port}>
          <Handle
            type="source"
            position={Position.Right}
            id={port}
            title={`Output ${portLabel(port)}`}
            style={{ top: portTop(kind) + i * 28 }}
          />
          {port !== "out" && (
            <span
              className="port-name output-port-name"
              style={{ top: portTop(kind) - 8 + i * 28 }}
            >
              {portLabel(port)}
            </span>
          )}
        </React.Fragment>
      ))}
      {data.share && (
        <>
          <Handle
            type="source"
            position={Position.Top}
            id="weights-out"
            isConnectable={false}
            style={{ left: "55%" }}
            className="weight-handle"
          />
          <Handle
            type="target"
            position={Position.Top}
            id="weights-in"
            isConnectable={false}
            style={{ left: "45%" }}
            className="weight-handle"
          />
        </>
      )}
      <div className="node-top">
        <span className="node-icon">
          <Icon size={16} />
        </span>
        <span className="node-kind">
          {kind === "loop"
            ? role === "denoising"
              ? "DENOISING LOOP"
              : "REPEAT / LOOP"
            : kind === "block" && role !== "composition"
              ? role.toUpperCase()
              : kind.replace("_", " ").toUpperCase()}
        </span>
        {expanded && (
          <span className="node-badge">
            {kind === "loop" ? `${p.iterations || 1}×` : "{}"}
          </span>
        )}
        {data._error && <CircleAlert size={15} className="error-icon" />}
      </div>
      <div className="node-title">
        {data.label}{" "}
        {data.annotation && (
          <span title="Has a design note" aria-label="Has a design note">
            ✎
          </span>
        )}
      </div>
      {data.share && (
        <div className="weight-badge" title={data.share}>
          ⛓ {data._groupLabel} · same weights{" "}
          {data.iteration && (
            <span>application {data.iteration.index + 1}</span>
          )}
        </div>
      )}
      <div
        className="node-body"
        style={
          laneNode
            ? {
                minHeight:
                  50 + Math.max(data._ports.length, data._outputs.length) * 28,
              }
            : undefined
        }
      >
        <Diagram kind={kind} p={p} body={data.body} role={role} />
        {kind === "input" && data._tokenizer && (
          <div className="input-tokenizer-summary">
            <Database size={12} />
            <span>{tokenizerLabel(data._tokenizer)}</span>
            <small>{format(data._tokenizer.capacity)} IDs</small>
          </div>
        )}
      </div>
      <div className="node-footer">
        <code>
          {shape
            ? `${data._outputs.length > 1 ? data._outputs.length + " × " : ""}[${shape.shape.join(", ")}]`
            : data._error
              ? "shape unresolved"
              : "connect input"}
        </code>
        <span>
          {data._count ? format(data._count) + " params" : shape?.dtype || ""}
        </span>
      </div>
      {expanded && (
        <button
          className="open-block nodrag"
          onClick={(e) => {
            e.stopPropagation();
            data._open(id);
          }}
        >
          Open{" "}
          {kind === "loop" ? "loop body" : kind === "moe" ? "experts" : "block"}
          <ArrowUpRight size={14} />
        </button>
      )}
    </div>
  );
}
export const nodeTypes = { architecture: ArchitectureNode };
export function ZoomReadout() {
  const { zoom } = useViewport();
  const flow = useReactFlow();
  return (
    <button
      className="zoom-readout"
      aria-label={`Zoom to 100 percent · current ${Math.round(zoom * 100)}%`}
      title="Zoom to 100% for readable node details"
      onClick={() => flow.zoomTo(1, { duration: motion(200) })}
    >
      {Math.round(zoom * 100)}%
    </button>
  );
}
