import { useState } from "react";
import { Plus, Trash2 } from "lucide-react";

function VariableValue({
  name,
  value,
  onChange,
}: {
  name: string;
  value: number;
  onChange: (value: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));
  const [error, setError] = useState("");
  function commit() {
    const next = Number(draft);
    if (!draft.trim() || !Number.isFinite(next)) {
      setError("Enter a finite number.");
      return;
    }
    setError("");
    if (next !== value) onChange(next);
  }
  return (
    <div>
      <input
        aria-label={`Value of $${name}`}
        aria-invalid={!!error}
        inputMode="decimal"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            commit();
          }
          if (event.key === "Escape") {
            event.stopPropagation();
            setDraft(String(value));
            setError("");
          }
        }}
      />
      {error && (
        <small className="field-error" role="alert">
          {error}
        </small>
      )}
    </div>
  );
}

export function GraphVariables({
  values,
  inherited,
  nested,
  compact = false,
  onChange,
}: {
  values: Record<string, number>;
  inherited: Record<string, number>;
  nested: boolean;
  compact?: boolean;
  onChange: (values: Record<string, number>) => void;
}) {
  const [name, setName] = useState("");
  const [value, setValue] = useState("128");
  const [error, setError] = useState("");
  function add(event: React.FormEvent) {
    event.preventDefault();
    const key = name.trim().replace(/^\$/, "");
    if (!/^[A-Za-z_][A-Za-z_0-9]*$/.test(key) || key === "sqrt_width") {
      setError(
        "Use a name such as hidden_width; sqrt_width is computed from width.",
      );
      return;
    }
    if (Object.hasOwn(values, key)) {
      setError(`$${key} already exists.`);
      return;
    }
    const number = Number(value);
    if (!value.trim() || !Number.isFinite(number)) {
      setError("Enter a finite number.");
      return;
    }
    onChange({ ...values, [key]: number });
    setName("");
    setError("");
  }
  const inheritedEntries = Object.entries(inherited).filter(
    ([name]) => !Object.hasOwn(values, name),
  );
  const tableValues = compact ? { ...inherited, ...values } : values;
  const [adding, setAdding] = useState(!compact);
  return (
    <section
      className={"graph-variables" + (compact ? " compact" : "")}
      aria-label="Graph variables table"
    >
      {!compact && (
        <p>
          Write <code>$name</code> in a block field to use a variable, or enter
          a number for an independent value.
          {nested &&
            " Values defined here apply inside this graph and override inherited values."}
        </p>
      )}
      <table>
        <thead>
          <tr>
            <th scope="col">Variable</th>
            <th scope="col">Value</th>
            <th scope="col">Actions</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(tableValues).map(([name, value]) => (
            <tr key={name}>
              <th
                scope="row"
                title={
                  Object.hasOwn(values, name)
                    ? undefined
                    : "Inherited · edit to override in this graph"
                }
              >
                <code>${name}</code>
              </th>
              <td>
                <VariableValue
                  key={`${name}:${value}`}
                  name={name}
                  value={value}
                  onChange={(value) => onChange({ ...values, [name]: value })}
                />
              </td>
              <td>
                <button
                  disabled={!Object.hasOwn(values, name)}
                  aria-label={`Delete $${name}`}
                  title={`Delete $${name}`}
                  onClick={() => {
                    const next = { ...values };
                    delete next[name];
                    onChange(next);
                  }}
                >
                  <Trash2 size={15} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {!Object.keys(tableValues).length && (
        <p className="field-help">No variables defined in this graph.</p>
      )}
      {compact && !adding && (
        <button className="small-button" onClick={() => setAdding(true)}>
          <Plus size={13} />
          Add variable
        </button>
      )}
      {adding && (
        <form className="variable-add" onSubmit={add}>
          <label>
            <span>Name</span>
            <input
              aria-label="New variable name"
              placeholder="hidden_width"
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <label>
            <span>Value</span>
            <input
              aria-label="New variable value"
              inputMode="decimal"
              value={value}
              onChange={(event) => setValue(event.target.value)}
            />
          </label>
          <button className="primary" type="submit">
            <Plus size={14} /> Add
          </button>
        </form>
      )}
      {error && (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
      {!compact && !!inheritedEntries.length && (
        <details>
          <summary>
            Inherited variables and block parameters ({inheritedEntries.length})
          </summary>
          <table>
            <tbody>
              {inheritedEntries.map(([name, value]) => (
                <tr key={name}>
                  <th scope="row">
                    <code>${name}</code>
                  </th>
                  <td>{value}</td>
                  <td>
                    <button
                      onClick={() => onChange({ ...values, [name]: value })}
                      aria-label={`Override $${name}`}
                    >
                      Override
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
      {!compact && (
        <p className="field-help">
          Only fields linked with <code>$name</code> follow these values.
          Removing a variable leaves its bindings visible as graph errors. Token
          inputs supply <code>$vocab</code> and, when available,{" "}
          <code>$mask_id</code> from their tokenizer.
        </p>
      )}
    </section>
  );
}
