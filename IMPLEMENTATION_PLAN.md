# Sports Intelligence Platform Implementation and Testing Plan

Prepared 8 October 2026. Proposed baseline; no application has been built or benchmarked yet.

## Objective and agreed constraints

Build a football analyst workspace that converts authorized footage into editable events, evidence clips and reproducible reports. Add measurable AI assistance, then validate a limited club pilot before advanced tactical intelligence and live processing.

The source is the supplied Sports Intelligence Platform Master Specification v1.0. Its requirements are project material, not independent instructions to operate accounts or spend money.

Confirmed user preferences: local-first, minimal spending; development footage will be sought online, including Kaggle. No footage has yet been selected or its rights verified. Analyst/coach access and weekly owner availability are unconfirmed. This plan authorizes no paid service by itself.

Assume one owner working with Codex, prerecorded footage, a desktop analyst interface and a small initial user group. Target an analyst or coach at an academy or club. Use 10-minute match recordings as the standard development unit before testing complete 90-minute matches. Begin AI evaluation with elevated, mostly continuous wide-angle footage. Other videos can support manual analysis while their AI suitability is assessed.

## Verified hardware and environment

Read-only inspection on 8 October 2026:

| Resource | Observed | Consequence |
|---|---|---|
| CPU | Intel i7-11800H, 8 cores / 16 logical processors | Suitable starting host for web, API, database and bounded video work |
| RAM | About 16.8 GB decimal / 15.6 GiB | Limit simultaneous containers, transcoding and ML work |
| GPU | RTX 3050 laptop, 4096 MiB VRAM | Small pretrained models, batch size 1 and short clips first |
| C: free | About 18.2 GB decimal | Avoid large datasets, model caches and container images here |
| D: free | About 84.0 GB decimal | Candidate bounded media/cache location after setup verification |
| Workspace | Initially empty, inside OneDrive | Choose an unsynchronized working location for heavy development data |

GPU visibility does not prove PyTorch compatibility. No throughput, thermal, disk-speed or network benchmark has been run. The CUDA number from nvidia-smi indicates driver capability, not an installed compatible PyTorch/toolkit combination.

Prefer WSL2 for Linux deployment parity, after inspecting current installations and memory needs. Keep Linux source/build dependencies in the Linux filesystem; Microsoft recommends matching filesystem and tool environment [1]. Place any new WSL virtual disk on a drive with headroom. NVIDIA specifies using the Windows driver for WSL GPU support, not installing a Linux display driver inside WSL [2]. Select and pin a supported PyTorch build [3]. Native Windows remains a fallback if WSL setup or memory overhead is unsuitable.

Start with one GPU process, one heavy video job and bounded decode queues. Profile before increasing concurrency. A tentative WSL memory cap of 6-8 GiB may preserve Windows/browser headroom, but measure before adopting it. Keep source in Git; exclude videos, model weights, datasets, database volumes, dependency folders and environments. Back up unique annotations separately. Do not move or delete existing files incidentally.

## Release sequence and estimates

R1 is the useful manual analyst product. R2 adds validated AI assistance. R3 is a limited reliable pilot. This separates the specification's manual Phase 1 from its broader AI-inclusive MVP.

| Stage | Implementation | Acceptance evidence | Focused effort |
|---|---|---|---|
| 0 Feasibility | Toolchain, footage/license shortlist, workflow, short decode/GPU benchmark | Reproducible environment, measured baseline, supported-footage envelope and R1 scope | 1-2 weeks |
| 1 Foundation | Repository, CI, migrations, organization/membership, identity integration and permissions | Clean setup; migration checks; two organizations isolated | 1-2 weeks |
| 2 Media | Validated upload, metadata, original/proxy storage, jobs and timestamp mapping | Supported files play; invalid/interrupted uploads recover; retries avoid duplicates | 1-2 weeks |
| 3 Workspace | Video, timeline, manual tagging/editing, filters, pitch, shortcuts and annotations | Code a 10-minute segment, reload it, edit events and navigate to their video moments | 2-3 weeks |
| 4 Outputs | Event statistics, accurate clips, report snapshots and export | Clips align; reports match hand-checked fixtures; permissions enforced | 1-2 weeks |
| 5 Perception | Integrate a football-fine-tuned detector and existing tracker, overlays, shot boundaries and team association | Baseline-versus-trained benchmark, versioned dataset/checkpoint, resource profile and safe unknown states | 2-4 weeks integration; training/labeling begins earlier |
| 6 Review | Restricted event candidates, confirmation/correction/rejection and audit history | Held-out event evaluation; timed review study; idempotent review | 2-3 weeks |
| 7 Spatial analytics | Calibration, coordinate mapping, coverage-aware heatmaps and qualified shape metrics | Independent calibration checks; invalid segments suppress metrics; versioned results | 2-4 weeks |
| 8 Pilot | Hosted option, resumable upload, quotas, operations, backup, access and usability | Restore drill; load/access tests; independent analyst sessions; cost per match measured | 2-4 weeks |

Application dependencies follow the stage order. A model-development workstream starts in Stage 0 and proceeds alongside the application stages, with local GPU experiments scheduled separately from heavy application/media workloads. Stages 0-4 total approximately 6-11 focused weeks; all stages total 14-26 before contingency. Reserve another 20-30% for integration, labels and corrections. These are planning estimates assuming focused owner participation, available footage and timely decisions, not delivery promises or predictions of model accuracy. Re-estimate after Stage 0 and R1. Part-time work extends calendar time. No expert tester means a portfolio prototype can proceed, but professional workflow validation remains incomplete.

## Architecture

Start with one web application, one modular API and CPU/GPU worker processes. Avoid a microservice for every domain feature.

Proposed stack: React/Next.js with TypeScript; FastAPI/Python; PostgreSQL and versioned migrations; Redis-backed job queue; FFmpeg; PyTorch/OpenCV; S3-compatible object storage in deployment; Docker and GitHub Actions. Pin exact versions after compatibility checks. Local storage may start with a filesystem adapter behind the same storage interface to reduce container load. Choose one identity integration and one deployment provider when needed.

```text
Browser analyst workspace
           |
        Web / API -------- PostgreSQL
           |                    |
   authorized media        durable job records
           |                    |
    Object storage <--- CPU / GPU workers <--- queue
           |
 original + proxy + clips + tracking + reports
```

The API authorizes media access and schedules jobs; it does not run long inference in HTTP requests. The browser streams a proxy. Durable job state lives in PostgreSQL. Workers publish validated artifacts atomically. A GPU worker can later move to a rented machine without redesigning the application.

Initial repository: apps/web, apps/api, workers/media, workers/vision, packages/contracts, packages/football, ml/evaluation, ml/experiments, infra, docs and tests/fixtures. Generate TypeScript API contracts from the backend schema. Keep experiments out of production request paths. Large fixtures have an external manifest, checksum and acquisition instructions.

## Data contracts before UI implementation

Core records: Organization, Membership, Team, Player, Competition, Season, Match, MatchPeriod, VideoAsset, Event, EventRevision, Annotation, Clip, ProcessingJob, ModelRun, Prediction, Review and ReportSnapshot.

Every owned record is organization-scoped. Enforce authorization in API, queries, storage and jobs; use database-level isolation as defense in depth where practical. Reject cross-tenant player/team/asset references. Define analyst, coach/viewer and administrator roles early, even before billing.

Separate source-media presentation time, proxy playback time and football match clock. Record mappings, explicit units, period/half, frame rate/time base and offsets. Handle halftime, stoppage time, extra time, replays and edited recordings. Do not assume video zero equals kickoff.

Specify pitch units, origin, axes, dimensions and attacking direction by period. Distinguish manual positions, measured/projected positions, interpolation and unavailable data. A tactical drawing is not proof of a measured real-world location.

Maintain immutable event revisions plus a current accepted view. Preserve original predictions, model/configuration versions and all review decisions. Keep provenance separate from verification status. Use optimistic concurrency and idempotency to prevent silent overwrites and duplicate approvals.

Track IDs are temporary and separate from player identities. Unknown identity is valid. Store dense tracking in partitioned columnar files with database metadata and browser-friendly downsampled views. Reports and metrics record input revisions and calculation versions. Corrections mark dependent outputs stale; past report snapshots remain reproducible.

## Media and job reliability

Validate file signatures, formats, sizes and duration server-side. Probe untrusted media in a restricted worker with time/memory/disk limits. Produce one compatible proxy first. Add adaptive delivery only when justified by measured needs.

Stream frames rather than saving every frame as an image. Use checkpointed processing segments and explicitly handle segment overlap, deduplication and track stitching. Camera cuts reset tracking/calibration assumptions. Replays must not silently double-count actions.

Jobs support queued/running/succeeded/failed/canceled states, heartbeat, progress, bounded retries and idempotency keys. Use temporary outputs and atomic final publication. Test worker termination, full disk, duplicate delivery, queue/storage outage and expired access URLs. Resume from valid checkpoints or safely restart.

Before a remote pilot, support resumable large uploads with checksum and completion validation. Deletion cancels associated jobs and covers originals and derivatives under a documented retention policy, including backup expiry. Keep the manual workspace responsive while AI runs.

## Dataset acquisition and annotation

Search Kaggle and original dataset publishers in Stage 0. For each candidate record original source, uploader, exact license, footage ownership/provenance, permitted research/training/display/redistribution/commercial use, annotation types, camera style, resolution and size. A Kaggle license label alone is not a verified rights chain; Kaggle provides its license information without warranties [9,10]. Prefer clear original-source terms and small subsets.

Use three data groups: authorized development footage; a frozen held-out evaluation set; and separately approved public portfolio-demo media. Permission for one use does not automatically establish another. If footage selection is delayed, generated timing videos and synthetic event fixtures let application work proceed, but cannot establish football AI accuracy.

SoccerNet provides relevant calibration/tracking/action tasks; some video access is gated and requires its NDA [4,5,11]. Inspect each task's terms before downloading. Do not assume benchmark data can ship with a commercial product.

Start with 3-5 short clips for pipeline debugging, then use 10-minute match recordings as the repeatable unit for annotation, training experiments and end-to-end tests. Select segments spanning normal play, camera movement, cuts, occlusions, restarts and difficult lighting rather than choosing only easy footage. Before model release, aim to expand toward segments from 10-20 independent matches across cameras/venues when accessible; this is a collection target, not proof of statistical sufficiency. Define labels before annotation: event timing, pass/shot boundaries, failures, possession transitions, occlusions and uncertainty. Densely label selected intervals and include negatives. Have a second annotator review a subset when available; disclose single-annotator limitations.

Split by whole match, grouping related recordings and replays together. Never randomly split adjacent frames across train/test. Hold out cameras/venues where possible. Freeze test manifests and tune only on training/validation. Report sample sizes and results by footage condition. Customer permission to process footage is separate from permission to reuse it for training.

## Model training and library reuse commitment

User clarification: training models and building on available libraries such as YOLO are core project requirements. Fine-tuning pretrained models is the default training strategy. Begin this work during feasibility and continue alongside the analyst application; do not postpone all training until the interface is complete. The manual application remains independently releasable while models are evaluated.

Reuse established implementations: a selected YOLO-family detector or suitable alternative; ByteTrack or BoT-SORT for tracking; OpenCV for image/geometry operations; PyTorch for model training; FFmpeg for media. Ultralytics documents pretrained training, checkpoint resume and configurable batch/image size [12], and integrations for ByteTrack/BoT-SORT [13]. Final model/version selection follows licensing, compatibility and local benchmark checks. Existing tracking algorithms usually need configuration and evaluation rather than detector-style retraining; any optional learned re-identification module is a separate model.

Training milestones:

1. Establish a frozen pretrained baseline on football footage before modifying weights.
2. Acquire usable annotations or label selected frames. Raw video alone is not a supervised detection dataset. Begin with player and ball classes; add referee/goalkeeper classes only when enough consistently labeled examples justify them.
3. Inspect boxes/class mappings, remove duplicates, include difficult and negative frames, and split by match before augmentation. Keep test matches isolated throughout iterative data collection.
4. Run a tiny training smoke test, including backward/optimizer steps, and an intentional small-subset overfit check to catch annotation or pipeline errors. This is a wiring test, not generalization evidence.
5. Fine-tune a small pretrained model. Use a short pilot run to measure minutes per epoch and peak training memory before scheduling longer runs. Select checkpoints/thresholds on validation data; use early stopping and a bounded experiment budget.
6. Compare the selected trained model with the baseline on held-out matches using identical preprocessing and metrics. Report ball and player results separately, along with speed and memory. If fine-tuning regresses, retain the baseline for deployment and record the failed experiment.
7. Inspect errors and expand labels selectively, especially for small/occluded balls and target camera conditions. Annotation proposals from a model must be reviewed; unreviewed predictions are not ground truth.
8. Version the dataset manifest, split, model weights, training configuration, code/dependency versions, seed, metrics and model card. Save both best-validation and resumable last checkpoints. Keep large weights outside Git.
9. Integrate the approved checkpoint with the product and preserve rollback to the prior model. Later train temporal event models or calibration/keypoint models as separate tasks when suitable labels exist; YOLO object detection alone does not learn tactical understanding.

Laptop training plan: first try a nano/small model, batch size 1, a bounded input size and supported mixed precision. Start with modest data-loader workers and avoid caching the full dataset in RAM. These are experiment settings, not a guarantee of fitting 4 GB VRAM. Training requires activations, gradients and optimizer state in addition to inference memory; benchmark training separately. Freezing some layers may be evaluated for transfer learning, but measure its effect. Do not reduce resolution until the ball becomes undetectable simply to fit memory.

Estimate training duration from measured epoch time multiplied by planned epochs, plus validation/checkpoint overhead. Use checkpoint/resume and monitor thermals. No full training run should begin without an estimated duration and storage footprint. Free hosted notebook GPUs may be considered if currently available under acceptable terms, but quotas and availability are not a dependable project requirement. Paid GPU time remains optional and needs a concrete cost decision if a quality-preserving training setup will not fit locally.

Training adds uncertain labeling and experiment effort. Earlier phase estimates are provisional and must be revised after dataset inspection and the first training benchmark; the Stage 5 integration estimate is not a promise that model research is complete in 2-4 weeks.

## AI implementation order

1. Small pretrained player detector on 30-60 second clips; benchmark end-to-end speed and correctness.
2. Ball detector evaluated separately; retain missing/occluded states. A small ball may require different resolution or modeling.
3. Within-shot tracking, with explicit resets at cuts and identity-switch measurement.
4. Team association with unknown classes and special handling for officials/goalkeepers.
5. Manual track-to-player association before attempting reliable automated identity.
6. Pitch calibration on supported views, validated on held-out landmarks and monitored during camera motion.
7. Limited pass/shot candidates using ball/team/temporal evidence. Bounding boxes alone do not establish football events.
8. Tactical hypotheses only when underlying measurements and expert-labeled definitions support them.

Keep object, identity, event and calibration confidence separate. Model scores are not calibrated probabilities unless evaluated as such. Projecting an airborne ball onto the pitch is not recovering its true 3D position. Camera discontinuities and calibration have dedicated football research because they are substantive problems [7].

Evaluate model-code and weight licenses before adoption. Ultralytics currently offers AGPL and Enterprise options [6]; decide intended source distribution and applicable obligations before choosing it. Compare alternative implementations and pretrained-weight terms. A separate worker is not a license exemption.

## Which analytics can ship when

| Feature | Required evidence | Treatment |
|---|---|---|
| Pass/shot counts, completion | Defined confirmed events | R1; show tagged duration/completeness |
| Shot maps, passing networks | Confirmed positions/player IDs | R1; distinguish partial tagging from full-match coverage |
| Possession | Explicit possession intervals or validated estimation | Do not equate pass share with possession |
| Heatmaps | Positions, time and coverage | Separate event maps from time-weighted tracking occupancy |
| Distance/speed | Continuous calibrated tracks and timestamps | Report observed intervals; do not bridge cuts/gaps as facts |
| Width/depth/compactness | Enough simultaneous team visibility | Suppress unsupported full-team claims from cropped views |
| Progressive actions/field tilt | Versioned spatial/event definitions | Publish denominator, direction and missing-data rules |
| xG | Appropriate validated model and permitted assets | Defer rather than invent an arbitrary score |
| Narrative insights | Trusted evidence and linked clips | Start with templates; evaluate any later language model |

Manual correction cannot recover off-camera measurements with certainty. Missing or unsuitable input should yield unavailable metrics, not invented precision.

## Computing and storage plan

Local development uses CPU for API/database/ordinary tests and GPU only for bounded vision work. Begin with one GPU worker, batch size 1 and short clips. Benchmark supported input scales such as 480/640/960 and multiple frame sampling rates. Record total decode-to-output speed, accuracy, peak RAM/VRAM, disk use and thermal behavior. Reducing sampling can miss short actions and destabilize tracks; compare quality before accepting it. Mixed precision/exported runtimes require equivalence checks.

Planning arithmetic, not measured performance: a 90-minute video sampled at 10 fps contains 54,000 frames. At an assumed 15 processed fps, the frame pipeline takes about 60 minutes; at 5 fps, about 180 minutes. Add transfer, proxy creation, model startup, exports and retries unless included in the benchmark.

Storage arithmetic: a 10-minute source at 8 Mbps is about 600 MB; a 2 Mbps playback proxy adds about 150 MB. Twenty 10-minute source/proxy pairs therefore use roughly 15 GB before clips, labels, models and backups. This fits the development strategy much better than twenty full matches, while still requiring bounded caches and free-space monitoring. A 90-minute source/proxy pair would use about 6.75 GB at those example bitrates. Do not save every decoded frame or auto-delete unique source material/annotations.

Treat each 10-minute segment as a dataset asset with its original match identifier, source start/end timestamps, checksum, rights record and processing version. Never split portions of the same match across training and held-out test sets. Longer-duration validation begins only after the 10-minute pipeline is stable: test 30 minutes, then one complete 90-minute match, checking memory, storage growth, track continuity, job recovery and total turnaround at each step.

Network arithmetic: 5.4 GB takes about 72 minutes at sustained 10 Mbps upload before overhead. Measure actual connectivity and interruption recovery before remote-first use.

Default recurring cloud budget is zero. Do not provision an always-on GPU. Optional cloud use is triggered only by demonstrated memory limits, unacceptable processing time, disruptive training or pilot demand. Benchmark the same clip on a short 16-24 GB GPU rental before committing. Runpod is one candidate with published prices; confirm live rates, availability, storage and transfer at booking [8]. No rental is necessary for the initial manual release.

Optional future planning envelopes, not quotations: $10-40/month for bounded experiments; $30-100/month for a small hosted app/database/storage/backup baseline, plus GPU usage. These are not the user's approved spending. For arithmetic only, 20 hours at an assumed $0.50/hour is $10 compute, excluding storage and transfer. Use job runtime/concurrency caps, spending alerts, storage quotas and shutdown checks. Track cost per successful match including failed runs.

Do not replace the laptop now. Consider compatible RAM/storage upgrades only after measured bottlenecks and concrete purchase comparisons; rented GPU time remains an alternative to buying hardware.

## Testing and acceptance gates

| Layer | Checks |
|---|---|
| Domain | Time/period/direction transforms, event validity, clip bounds and hand-computed metric fixtures |
| API/database | Real PostgreSQL integration, constraints, migrations, pagination, concurrent edits and idempotency |
| Security | Cross-tenant records/media/exports/jobs denied; role restrictions; revoked users; expired access; no secrets in logs |
| Browser | Login, upload, play, tag, edit, filter, annotate, export, reload; keyboard and error-state behavior |
| Media | Constant/variable frame rate, long-GOP seek, missing audio, rotation, corruption and period/replay mapping |
| Workers | Kill/restart, duplicate delivery, cancellation, storage outage, full disk and partial artifacts |
| ML | Frozen held-out data, condition-specific metrics, unknown cases and versioned reproducibility |
| Analytics | Reject unconfirmed data by default, invalidate stale results, label coverage, reproduce snapshots |
| Performance | Initial 3 concurrent analysts and one heavy job; API latency, playback startup, UI, memory and storage |
| Operations | Restore into clean environment, staged migration, model/application rollback and deletion/export checks |

Proposed timing gate: a generated video with known frame markers proves persisted time mappings and re-encoded clip boundaries within one output frame. Target practical browser seeking within 100 ms on declared supported proxies. These are acceptance targets, not achieved claims; narrow unsupported media contracts if needed.

Proposed responsiveness targets: immediate local event feedback under 100 ms and ordinary metadata API p95 under 500 ms under a documented small-pilot workload, excluding uploads/inference. Measure actual hardware/data/concurrency.

ML reporting: separate player/ball precision and recall; tracking IDF1/HOTA and identity switches; team assignment with unknowns; calibration error and valid coverage; event precision/recall with explicit timestamp tolerance and deduplication. Report sample counts and uncertainty where supportable.

Provisional AI review gate: at least 85% event precision on the declared supported slice, with recall disclosed, and at least 20% lower median analyst time at comparable final annotation quality. Validate through at least 3 analysts on matched clips where available, varying task order to reduce learning effects. Small studies are directional. If professional testers are unavailable, do not claim expert validation. Set detector/calibration thresholds after baseline work identifies downstream needs. A failed AI gate leaves the module experimental without blocking the useful manual product.

Each change runs relevant lint/type/static checks, unit and affected integration tests; critical workflows run browser E2E checks. CI uses small synthetic/permitted fixtures and CPU-compatible tests. Run a real GPU smoke test and full fixed benchmark before model promotion or relevant pipeline changes. Mocked model output verifies app plumbing, not AI quality.

Each milestone retains the change, acceptance results, demo, benchmark updates and limitations. Passing automated checks does not replace observed analyst use.

## Pilot, deployment and portfolio

Keep staging/production secrets, databases and buckets separate. Introduce hosted operation only when local workflows are stable and remote testers justify the cost. Limit organizations, upload sizes, retention and concurrent jobs. Portfolio media must be distributable; use a read-only/resettable demo isolated from real customer data.

Observe job failures/duration, queue age, API errors, storage growth, memory and per-match cost. Correlate logs using request/job IDs without exposing footage or tokens. Set provisional recovery targets, such as a maximum 24-hour data-loss window and four-hour restore, then verify before promising them. Back up both database and necessary media; test a real restore.

Pilot completion requires independent upload-to-report sessions, no critical access/data-loss defects, usable turnaround, measured costs, documented rights/retention, backup recovery and evidence of user value. Introduce billing only when users, pricing and accurate usage accounting are established.

Portfolio deliverables: runnable local application, permitted sample/demo, architecture diagram, installation guide, migration/CI evidence, reproducible benchmark with honest failures, short walkthrough and a case study showing why human corrections matter. Public display can use recorded demos before recurring hosting is affordable.

## Later capabilities

Advanced pressing/transitions/overloads require tactical definitions, reliable positions/events and expert-reviewed sequences. Research one capability at a time with a bounded experiment and go/no-go decision. No fixed success date is credible before baseline data exists.

Language summaries start with templates over verified events. A later model must cite evidence IDs and clips; check numerical statements and allow insufficient-evidence results. Evaluate hallucinations and citation correctness. A large local language model is not required initially.

Live operation follows stable batch processing. Implement stream recovery, late/out-of-order events, rolling jobs and a measured latency/cost budget. Replay recordings at real time first. Batch speed alone does not establish live readiness.

Additional sports reuse media, organization, job and provenance infrastructure, while keeping sport-specific rules, geometry, models and event semantics separate. Enterprise APIs, federation scale and hardware partnerships follow validated demand and dedicated capacity plans.

## Working process and first execution package

Codex implements bounded features, tests, reproducible benchmarks, debugging, migrations and documentation. The owner supplies priorities, authorized footage, football interpretation, tester access and spending decisions. Model capability accelerates implementation; it cannot replace ground truth, GPU memory or user evaluation.

Every task includes a user outcome, dependencies, scope and observable acceptance checks. Build a small vertical slice, inspect it in the running application, run the relevant tests and record limitations. Re-estimate when evidence changes assumptions.

First implementation package:

1. Inspect toolchain/WSL/container state and storage; establish a reproducible environment.
2. Shortlist online datasets and verify terms; obtain a small permitted sample and a synthetic timing fixture.
3. Benchmark decode/proxy creation and a pretrained detector, then run a tiny fine-tuning/backpropagation smoke test on labeled football frames; log inference and training resources separately.
4. Choose the initial footage contract and code/model licensing direction.
5. Establish repository, pinned environments, CI and architecture decisions.
6. Define organization, match, media, event/revision and time contracts with fixtures.
7. Build a persistent slice: open a video, create an event, save, reload and click the event to return to its moment.

This package proves the foundation. It does not claim complete commercial readiness or automated football understanding.

## Sources checked

[1] Microsoft WSL filesystem guidance: https://learn.microsoft.com/en-us/windows/wsl/filesystems
[2] NVIDIA CUDA on WSL: https://docs.nvidia.com/cuda/wsl-user-guide/
[3] PyTorch installation selector: https://pytorch.org/get-started/locally/
[4] SoccerNet camera calibration: https://www.soccer-net.org/tasks/camera-calibration
[5] SoccerNet challenges: https://www.soccer-net.org/challenges
[6] Ultralytics licensing guidance: https://www.ultralytics.com/license
[7] Cioppa et al., camera calibration and player localization: https://openaccess.thecvf.com/content/CVPR2021W/CVSports/papers/Cioppa_Camera_Calibration_and_Player_Localization_in_SoccerNet-v2_and_Investigation_of_CVPRW_2021_paper.pdf
[8] Runpod pricing: https://www.runpod.io/pricing
[9] Kaggle dataset documentation: https://www.kaggle.com/docs/datasets
[10] Kaggle license disclaimer: https://www.kaggle.com/page/license-disclaimer
[11] SoccerNet data access: https://www.soccer-net.org/data

Sources support particular environment, research, licensing and provider facts. Architecture, timing estimates, budgets and quality gates are proposed engineering judgments, not vendor guarantees or measured product results.

[12] Ultralytics model training: https://docs.ultralytics.com/modes/train
[13] Ultralytics tracking: https://docs.ultralytics.com/modes/track


