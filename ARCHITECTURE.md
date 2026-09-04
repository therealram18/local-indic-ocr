# Architecture

Two models with a JSON file between them.

```text
page image ──> IndicDocLayout ──> layout JSON ──> IndicBlockOCR ──> page JSON + markdown
```

`IndicDocLayout` finds the blocks and orders them. `IndicBlockOCR` transcribes one block crop
per request. Everything between the two models (cropping, prompt selection, reassembly) is this
package.

## The call path

One page, top to bottom, with the file that owns each step.

```text
IndicOCR.parse(image)                              idp_offline.py

  IndicDocLayout.detect(image)                           idp_offline.py
    _open                     image -> PIL RGB           idp_offline.py
    IndicDocLayoutBackend.detect                         idp_layout.py
      infer                   -> [y0,x0,y1,x1] at 0-1000 idp_model_infer.py
      convert + clamp_to_page -> pixel [x0,y0,x1,y1]     idp_layout.py, blocks.py
      clean_layout            drop duplicate boxes       idp_blocks.py
      _densify                order -> gap-free 0..n-1   idp_layout.py
    = PageResult, every block has text None              idp_types.py

  IndicBlockOCR.run(image, layout)                       idp_offline.py
    _as_page                  dict -> PageResult, validated
    is_transcribed            drop OCR_SKIP_LABELS       idp_contract.py
    resolve_nested_equations  drop nested equations      idp_blocks.py
    build_requests            -> [CropRequest]           idp_recognizer.py
      crop_for + area_clamp   block -> image             idp_crops.py
      prompt_for(block.type)  type  -> prompt            idp_contract.py
    backend.transcribe        [CropRequest] -> [str]     a recognizer backend
    match by order            texts -> blocks            idp_offline.py
    reconstruct               blocks -> markdown         idp_reconstruct.py
  = PageResult, every block has text
```

Blocks that were skipped are not deleted. They come back with `text: ""`.

## Files by role

**Contract and data.** The vocabulary everything else shares.

| file | holds |
| --- | --- |
| `idp_contract.py` | prompts, the label to type map, which labels are skipped |
| `idp_types.py` | `Block`, `PageResult`, and every tunable config |

**The layout model.** Torch lives here and nowhere else.

| file | holds |
| --- | --- |
| `idp_model_infer.py` | preprocessing and decode for one page |
| `idp_model_labels.py` | the 37 classes, pure Python |
| `idp_model_ppdoc.py` | the PP-DocLayoutV3 subclass |
| `idp_model_order_loss.py` | reading-order decode, training loss |

**Geometry and crops.** No model, no network.

| file | holds |
| --- | --- |
| `idp_blocks.py` | box math, `clean_layout`, nested-equation dedup |
| `idp_crops.py` | `crop_for` and `area_clamp` |

**Backends.** Two Protocols, five implementations, one per deployment.

| class | file | used by |
| --- | --- | --- |
| `IndicDocLayoutBackend` | `idp_layout.py` | the real detector |
| `JsonLayoutBackend` | `idp_layout.py` | replaying a layout file, no torch |
| `HfRecognizer` | `idp_recognizer.py` | plain transformers, the Hub package |


**Orchestration.**

| file | holds |
| --- | --- |
| `idp_offline.py` | `IndicDocLayout`, `IndicBlockOCR`, `IndicOCR` |
| `idp_recognizer.py` | `CropRequest`, `build_requests`, the recognizer Protocol |
| `idp_reconstruct.py` | blocks to markdown, math and hyphen repair |

## Invariants

1. **Two box conventions.** The model emits `[y0, x0, y1, x1]` normalised to 0-1000. The
   pipeline uses pixel `[x0, y0, x1, y1]`. 
2. **`order` must be gap-free and 0-based.** Transcriptions are matched back to blocks by
   position, so a gap or a duplicate moves text onto the wrong block.
3. **The crop clamp is on area, not on a side.** Pinning a side explodes elongated crops. See
   `CropConfig` in `idp_types.py`.
4. **`type` selects the prompt.** A wrong `type` changes what the model was asked to do, not
   just how a block is labelled. Unrecognised labels are rejected for this reason.
