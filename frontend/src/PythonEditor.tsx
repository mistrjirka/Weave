import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { basicSetup } from "codemirror";
import { Annotation, EditorState } from "@codemirror/state";
import { EditorView, keymap } from "@codemirror/view";
import { python } from "@codemirror/lang-python";
import {
  HighlightStyle,
  indentUnit,
  syntaxHighlighting,
} from "@codemirror/language";
import { tags } from "@lezer/highlight";

const externalChange = Annotation.define<boolean>();
const syntax = HighlightStyle.define([
  {
    tag: [tags.keyword, tags.controlKeyword],
    color: "var(--theme-violet-text, #70518e)",
  },
  {
    tag: [tags.string, tags.special(tags.string)],
    color: "var(--theme-green-text, #386d50)",
  },
  {
    tag: [tags.number, tags.bool, tags.null],
    color: "var(--theme-amber-text, #86611b)",
  },
  {
    tag: tags.comment,
    color: "var(--theme-muted, #59685d)",
    fontStyle: "italic",
  },
  {
    tag: [tags.function(tags.definition(tags.variableName)), tags.className],
    color: "var(--theme-blue-text, #3f6c91)",
  },
  {
    tag: [tags.operator, tags.meta],
    color: "var(--theme-coral-text, #9f4737)",
  },
]);
const theme = EditorView.theme({
  "&": {
    backgroundColor: "var(--theme-input, #fcfdfb)",
    color: "var(--theme-text, #263b30)",
    fontSize: "12px",
    height: "560px",
  },
  ".cm-scroller": {
    overflow: "auto",
    fontFamily: "'JetBrains Mono', monospace",
    lineHeight: "1.7",
  },
  ".cm-content": {
    padding: "12px 0",
    caretColor: "var(--theme-text, #263b30)",
  },
  ".cm-gutters": {
    backgroundColor: "var(--theme-base, #f1f4ef)",
    color: "var(--theme-muted, #59685d)",
    borderRight: "1px solid var(--line)",
  },
  ".cm-activeLine, .cm-activeLineGutter": {
    backgroundColor:
      "color-mix(in srgb, var(--theme-green-accent, #608d76) 9%, transparent)",
  },
  ".cm-selectionBackground, &.cm-focused .cm-selectionBackground": {
    backgroundColor:
      "color-mix(in srgb, var(--theme-blue-text, #3f6c91) 24%, transparent) !important",
  },
  ".cm-cursor": { borderLeftColor: "var(--theme-text, #263b30)" },
  ".cm-panels, .cm-tooltip": {
    backgroundColor: "var(--theme-raised, #fff)",
    color: "var(--theme-text, #263b30)",
    border: "1px solid var(--line)",
  },
});

export interface PythonEditorHandle {
  insertSnippet: (text: string) => void;
}
interface Props {
  value: string;
  onChange: (value: string) => void;
  onSave: () => void;
}

export const PythonEditor = forwardRef<PythonEditorHandle, Props>(
  function PythonEditor({ value, onChange, onSave }, ref) {
    const parent = useRef<HTMLDivElement>(null);
    const view = useRef<EditorView | null>(null);
    const callbacks = useRef({ onChange, onSave });
    const initialValue = useRef(value);
    useEffect(() => {
      callbacks.current = { onChange, onSave };
    }, [onChange, onSave]);
    useEffect(() => {
      if (!parent.current) return;
      const editor = new EditorView({
        parent: parent.current,
        state: EditorState.create({
          doc: initialValue.current,
          extensions: [
            keymap.of([
              {
                key: "Mod-s",
                run: () => {
                  callbacks.current.onSave();
                  return true;
                },
              },
            ]),
            basicSetup,
            python(),
            indentUnit.of("    "),
            theme,
            syntaxHighlighting(syntax),
            EditorView.contentAttributes.of({
              "aria-label": "Training Python script",
              "aria-multiline": "true",
              spellcheck: "false",
            }),
            EditorView.updateListener.of((update) => {
              if (
                update.docChanged &&
                !update.transactions.some((transaction) =>
                  transaction.annotation(externalChange),
                )
              )
                callbacks.current.onChange(update.state.doc.toString());
            }),
          ],
        }),
      });
      editor.scrollDOM.tabIndex = 0;
      editor.scrollDOM.setAttribute("role", "region");
      editor.scrollDOM.setAttribute("aria-label", "Python editor scroll area");
      view.current = editor;
      return () => {
        editor.destroy();
        view.current = null;
      };
    }, []);
    useEffect(() => {
      const editor = view.current;
      if (editor && editor.state.doc.toString() !== value)
        editor.dispatch({
          changes: { from: 0, to: editor.state.doc.length, insert: value },
          annotations: externalChange.of(true),
        });
    }, [value]);
    useImperativeHandle(
      ref,
      () => ({
        insertSnippet(text) {
          const editor = view.current;
          if (!editor) return;
          const selection = editor.state.selection.main;
          const line = editor.state.doc.lineAt(selection.from);
          const indent = /^\s*/.exec(line.text)?.[0] ?? "";
          const prefix = editor.state.doc.sliceString(
            line.from,
            selection.from,
          );
          const inline = prefix.trim().length === 0;
          const position = inline ? selection.from : line.to;
          const inserted =
            (inline ? "" : "\n" + indent) +
            text.trimEnd().replaceAll("\n", "\n" + indent) +
            "\n" +
            indent;
          editor.dispatch({
            changes: {
              from: position,
              to: inline ? selection.to : position,
              insert: inserted,
            },
            selection: { anchor: position + inserted.length },
            scrollIntoView: true,
          });
          editor.focus();
        },
      }),
      [],
    );
    return <div className="training-code-editor" ref={parent} />;
  },
);
