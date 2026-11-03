# Component-library review

Weave currently uses React 19, custom CSS and shared theme variables. React Flow renders the neural-network canvas; CodeMirror renders the Python editor. There is no general-purpose component library for forms and dialogs.

The validation-dialog regression came from a portal: the dialog was rendered outside `.training-workspace`, which supplied the controls' appearance. Its secondary button inherited the global transparent-button reset, while its primary button kept a separate global style. The fix shares the training control styles with portaled dialogs, including field heights, borders, colors, mobile text sizes and focus states. Dialog padding and scroll ownership are explicit.

## Candidates

| Candidate | What it provides | Fit for Weave |
| --- | --- | --- |
| [Radix Themes](https://www.radix-ui.com/themes/docs/components) | Styled React components, including buttons, fields, selects and dialogs, with a shared [theme/token system](https://www.radix-ui.com/themes/docs/theme/overview). | Best candidate for a small first trial. Adopt one complete dialog at a time and map its theme to Weave's light, Slate and Black palettes. No Tailwind migration is needed. |
| [Mantine](https://mantine.dev/getting-started/) | A broad React component library with configurable sizes and themes. Its [Styles API and CSS imports](https://mantine.dev/styles/mantine-styles/) support existing CSS projects. | Also a strong option, especially if training forms grow more complex. Trial the same form before choosing between it and Radix Themes. |
| [shadcn/ui](https://ui.shadcn.com/docs) | Components copied into the application so their source can be customized. The current [manual setup](https://ui.shadcn.com/docs/installation/manual) uses Tailwind. | More setup for this repository, which currently uses plain CSS. Owning the component source is useful, but requires maintaining a consistent local component set. |

The recommendation is an assessment of this repository, not a benchmark of bundle size or rendering speed. No component library has been installed in this change.

## Proposed trial

Implement the validation dialog in Radix Themes and Mantine separately, using the same labels, state transitions and dimensions. Compare desktop and phone screenshots, keyboard navigation, focus restoration, dark palettes, validation errors and pending/disabled states. Measure the resulting production bundles before choosing.

After choosing, migrate buttons, text/number fields, selects, dialogs and confirmation footers together within each window. Set one spacing scale and control-size convention, with at least 44px touch targets on phones. Keep React Flow, CodeMirror and the graph's operation colors. Avoid mixing the old and new form controls in the same dialog.

Functional work takes priority: live validation acknowledgement, checkpoint restoration, script-edit protection, sample freshness and dashboard loading must remain covered during any visual migration.
