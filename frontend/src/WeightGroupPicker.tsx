import { Plus } from "lucide-react";
import type { WeightGroupChoice } from "./weightGroups";

export function WeightGroupPicker({
  value,
  choices,
  onChange,
  onCreate,
}: {
  value: string;
  choices: WeightGroupChoice[];
  onChange: (value: string) => void;
  onCreate: () => void;
}) {
  return (
    <section className="weight-group-picker" aria-label="Shared weights">
      <label className="property-field">
        <span>Weight group</span>
        <select
          aria-label="Weight group"
          value={value}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">Independent weights</option>
          {choices.map((choice) => (
            <option
              key={choice.id}
              value={choice.id}
              disabled={!choice.compatible}
            >
              {choice.label}
            </option>
          ))}
        </select>
      </label>
      <button className="small-button" onClick={onCreate}>
        <Plus size={14} />
        New weight group
      </button>
      <p className="field-help">
        Only groups for the same node type and block interface are offered.
        Joining a group uses its existing settings; later edits update all its
        applications.
      </p>
    </section>
  );
}
