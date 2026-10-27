# Third-party architecture references

The architecture-specific computations in `backend/runtime.py` are ports of published mathematical operations from Cactus Compute's Needle architecture. The saved upstream source and license are in `references/needle3_architecture.py` and `references/needle_LICENSE`. Preserve applicable Apache-2.0 attribution when redistributing derived code.

`references/thimble_model.py` is an unmodified upstream reference used for parity tests. Its MIT license and copyright notice are in `references/thimble_LICENSE`.

Reference model configurations come from Cactus-Compute's Hugging Face repositories. Revisions and links are included with each project. No pretrained model weights are bundled.

LLaDA's published sampling implementation informs the masked-text sampling example. This example is an independent composable implementation, not a copy of the LLaDA checkpoint or a complete reproduction of its generation API.

Frontend libraries retain their respective licenses in installed package metadata. DM Sans and Manrope are distributed under the SIL Open Font License.

## ELK layout engine

The frontend bundles unmodified elkjs 0.12.0 from the Eclipse Layout Kernel project, including its browser worker. It is used under the Eclipse Public License 2.0. The package license is preserved in `references/elkjs_LICENSE.md`. Corresponding source and upstream notices: https://github.com/kieler/elkjs/tree/v0.12.0 and https://github.com/eclipse-elk/elk. The layout adapter and React Flow renderer are separate application code.
