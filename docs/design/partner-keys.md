# Design: partner-owned LLM keys in hosted Moose

Status: phases 1 (vault and tokens), 2 (Keycloak accounts and dashboard) and 4
(shared models without LiteLLM) implemented; phase 3 open.

## Goal

External users use the hosted Moose without deploying it, with their own LLM
provider keys. Each user has an account and manages their own keys in a
dashboard. As the platform operator, we must not be able to:

- read a user's provider keys,
- use a user's models (spend on their keys), or
- read a user's annotation results.

## What is and is not achievable

A server that calls an LLM provider on a user's behalf holds that user's key in
plaintext while it makes the call. No server-side design can make it impossible
for an operator who changes the server code to capture keys in flight.

This design therefore guarantees the following, and states the remainder as a
trust assumption:

| Guarantee | Holds against |
| --- | --- |
| Stored keys and results cannot be decrypted by the operator | database access, backups, admin UIs, a stolen database |
| The operator cannot use a user's keys without the user | jobs only run when the user's own token unlocks the keys |
| Keys exist in plaintext only in memory, only during the user's request or job | logs, job records, crash dumps of the database |
| **Not** guaranteed: a modified, malicious server capturing keys in flight | requires trusting the running code |

Users shrink the remaining risk on their provider's side: a key created only
for Moose, with a small budget, only the models they want, an expiry date, and
revocable at any time (on OpenRouter, a key with a guardrail).

Users who need full zero-knowledge use the Python library (`pip install
moose`), where LLM calls run on their own machine and Moose never sees the key.

## Why not store partner keys in LiteLLM

LiteLLM encrypts stored provider keys with a server-held salt key. The operator
holds that key and can decrypt every stored key or route requests through any
stored model. That fails the goal. LiteLLM remains an option for the models the
operator provides (see [Shared models](#shared-models)).

## Design

### Overview

```text
            Keycloak (accounts, login, roles)
                 |  OIDC
                 v
 user ---> Moose dashboard / API ---> per-user vault (encrypted in MongoDB)
                 |                         |
                 |   user's API token unlocks the vault in memory
                 v                         v
           job queue (in memory) ---> provider (OpenRouter, ...) with user's key
```

### Accounts: Keycloak

User accounts, login, password reset and roles come from the Keycloak instance
already running on the VM, through OpenID Connect. Moose does not store
passwords. Roles:

- `moose-admin`: manages shared schemas and shared models; has **no** access to
  any user's vault.
- `moose-user`: manages their own keys, tokens and jobs.

Because Keycloak handles the login, Moose never sees the user's password, so
the password cannot unlock the vault. Unlocking uses Moose API tokens instead.

### Per-user vault: envelope encryption

Each user has one random 256-bit **data key**. Everything secret of that user
is encrypted with it using AES-256-GCM:

- provider keys (OpenRouter, other OpenAI-compatible providers),
- optionally, job payloads and results.

The data key itself is never stored in plaintext. It is stored only **wrapped**
(encrypted) once per unlock credential:

| Unlock credential | Wrapping key | Used by |
| --- | --- | --- |
| Moose API token (high entropy, shown once) | HKDF-SHA256 of the token | API clients and the dashboard session |
| Recovery code (shown once at vault creation) | HKDF-SHA256 of the code | regaining access after losing all tokens |

MongoDB holds per user: the wrapped data keys, each token's SHA-256 hash (for
authentication), and the encrypted secrets. None of this can be decrypted
without a token or recovery code, which only the user has.

### API tokens: authentication and unlock in one

1. On the user's first login, Moose creates their vault: a new data key, a
   first API token and a recovery code, each shown once and stored only as a
   hash plus a wrapped copy of the data key. Later, while the vault is
   unlocked, the user can create more tokens (one per script or machine) the
   same way.
2. Each request sends the token (`X-API-Key`). Moose finds the user by the token
   hash, derives the wrapping key from the token, unwraps the data key, and
   decrypts what the request needs, all in memory.
3. Revoking a token deletes its hash and wrapped data key. Revoking every token
   and the recovery code leaves the vault permanently unreadable, by design.

### Dashboard session

The dashboard needs the data key while the user works. After the Keycloak
login, the user unlocks the vault once per session with a token or the recovery
code. The data key stays in server memory for that session (idle timeout, for
example 30 minutes) and is never written anywhere.

The current Streamlit UI runs on our server, so this is within the same trust
assumption. A browser-only UI that calls providers directly would remove it,
but is a larger change and not part of this proposal.

### Jobs

- At submit, the request's token unlocks the data key; the job's provider key
  is decrypted and handed to the in-memory job queue, as partner jobs already
  work today. Nothing secret enters MongoDB in plaintext.
- If job encryption is on, payload and result are encrypted with the data key
  before storage; `GET /jobs/{id}` decrypts them with the caller's token.
- A restart before a job finishes fails it with a request to resubmit.
- Job metadata (model used, tokens, cost) stays in plaintext so users and the
  operator can see usage, but it contains no content.

### Provider endpoints

Users may only choose providers from a fixed allowlist of base URLs (for
example OpenRouter, OpenAI, Anthropic, Mistral). A user-supplied URL would let
anyone make our server send requests into the internal network.

### Shared models

Models the operator provides (and pays for) are separate from user keys:

- **Option 1:** keep LiteLLM for shared models only.
- **Option 2:** drop LiteLLM; shared models use the operator's own OpenRouter
  key directly, with OpenRouter guardrails for allowlists and budgets.

Option 2 removes two containers and most of the gateway quirks. Either way, the
model a job used is recorded in its metadata.

**Decision (phase 4):** option 2. The operator's key is set on the dashboard's
Admin page and stored encrypted with a server key (`MOOSE_SECRET_KEY`), because
jobs of any user must be able to use it without the operator present. The
operator picks the shared models from OpenRouter's catalogue; spending is capped
with a credit limit on the key at the provider.

## What the operator can and cannot do

| Action | Possible? |
| --- | --- |
| See which users exist, their roles, token creation dates | yes (Keycloak, Moose metadata) |
| See job metadata: model, tokens, cost, timestamps | yes |
| Read a user's provider keys | no |
| Run a job on a user's key without the user | no |
| Read a user's job payloads and results | no, if job encryption is on |
| Recover a user's vault after they lose all tokens and the recovery code | no, by design |
| Capture keys by deploying modified code | yes; this is the trust assumption |

## Changes to the current code

- Replace the partner path that accepts LiteLLM virtual keys with vault-backed
  API tokens. Keep the admin key and admin-only schema routes.
- Add `moose.vault`: data keys, wrapping, AES-GCM, using the `cryptography`
  package.
- Add vault and token endpoints (`/vault/*`, `/tokens/*`) and dashboard pages:
  my keys (masked), my tokens, unlock.
- Add Keycloak OIDC login to the dashboard.
- Keep in-memory partner jobs and per-job model metadata as they are.

## Phases

1. **Vault and tokens, without Keycloak.** Encrypted per-user key storage, token
   unlock, jobs on user keys, tests showing the operator cannot decrypt. Local
   only.
2. **Keycloak accounts and dashboard pages.**
3. **Encrypted job payloads and results.**
4. **Shared models:** decide LiteLLM or direct OpenRouter, and remove the unused
   path.

## Open questions

1. Is the trust assumption (operator does not deploy malicious code) acceptable,
   or must some users get full zero-knowledge through the library only?
2. Should job payloads and results be encrypted by default, given that the
   operator then cannot help debug a user's failed job?
3. Which providers should the allowlist include beyond OpenRouter?
4. Shared models: keep LiteLLM or switch to direct OpenRouter?
5. Does SINTEF need a data processing agreement with each partner for this
   setup, given that annotated text may contain personal data?
