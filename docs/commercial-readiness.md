# Commercial Readiness Gate

This document is an engineering checklist, not legal advice.

A locally successful fine-tune is not automatically a commercially deployable voice. Treat the following as release gates:

## Rights and provenance

- Record the owner, permitted uses, territories, duration, and withdrawal process for every voice.
- Store hashes of source recordings and consent documents in an access-controlled system.
- Review the source-code license, pretrained checkpoint license, audio-tokenizer license, dataset terms, and third-party model terms independently.
- Do not infer commercial permission from the repository source license alone.

## Safety and access control

- Add authentication, RBAC, tenant isolation, rate limits, quotas, and audit logs.
- Restrict model export and voice-clone prompt download privileges.
- Log who uploaded, trained, generated, exported, and deleted each voice asset.
- Add a process for complaints, consent withdrawal, incident response, and model revocation.
- Consider provenance metadata or watermarking for generated audio.

## Quality and release testing

- Run fixed Thai and multilingual pronunciation suites on every candidate checkpoint.
- Measure intelligibility, speaker similarity, naturalness, silence, duration errors, and long-form stability.
- Compare candidate checkpoints against the base model with blind listening tests.
- Keep a reproducible run manifest containing code revision, model revision, config hash, dataset fingerprint, and evaluation outputs.
- Promote a checkpoint only through an explicit release step; never treat the latest checkpoint as automatically best.

## Operations

- Containerize the runtime and pin model/dependency revisions.
- Test stop, resume, disk-full, failed download, corrupted audio, and interrupted checkpoint scenarios.
- Monitor GPU memory, queue depth, latency, failure rate, disk growth, and generated-audio abuse signals.
- Back up consent records and release metadata separately from disposable training checkpoints.
