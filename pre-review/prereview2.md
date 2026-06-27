## Pre-review by Sanjana Kalarickal, A18224149
## Review of TritonLiving

### 1. Project summary/implementation

#### a. Summary

This is an **auth + (planned) live-URL** project: a self-built, full-stack Next.js web app. The intended users are **UCSD students looking for roommates** (sign-in is gated to `@ucsd.edu` Google accounts). The tool collects a student's housing and lifestyle preferences through an onboarding wizard, then shows a ranked feed of compatible roommate candidates. Each recommendation has a compatibility percentage, a short list of "why we recommend" reasons, and "potential differences" to discuss. Users can save matches, hide matches, and message people one-on-one.

#### b. Demo attempt

The `README.md` has clear run instructions, so I followed those. The setup it asks for is heavier than a one-command demo: Node 20+, a local PostgreSQL database, Google OAuth client credentials, and a hand-written `.env.local` before anything works.
SYSTEM consists of :
1. **Entry point:** `src/app/dashboard/page.tsx` → on load it calls `fetch("/api/recommendations")`. (If that returns 400, it redirects the user to `/onboarding`.)
2. **API handler:** `src/app/api/recommendations/route.ts` → `getCurrentUser()` (from `src/backend/auth.ts`), then `getRecommendationsForUser(user.id)`.
3. **Engine, `src/backend/recommendations/index.ts`:**
4. **Render:** the dashboard shows one card at a time — avatar, name, score %, budget/location/room-type summary, "Why we recommend," "Potential differences," and Hide / Favorite buttons.


#### c. Proposal component check:

1. **UCSD-only Google OAuth.** The proposal points to `src/lib/auth.ts`. The code matches the *claim* but **not the path** — it actually lives in `src/backend/auth.ts`. The `signIn` callback checks `user.email.endsWith("@ucsd.edu")` and otherwise redirects to `/access-denied`.

2. **One-on-one messaging.** Proposal claims `Conversation`/`ConversationParticipant`/`Message` models with UI in `src/app/messages/` and APIs in `src/app/api/conversations/`. These all exist (`src/app/messages/page.tsx`, `src/app/messages/[userId]/page.tsx`, `src/app/api/conversations/route.ts`, `src/app/api/conversations/[id]/messages/route.ts`). Matches as written.

A "no longer planned" item: **SQLAlchemy / JWT / AWS S3 / Pytest** were all dropped (they switched to Prisma + NextAuth + data-URL image storage). This is reasonable for the final deadline — they pivoted from a Python stack to TypeScript/Next.js, so a Python ORM and test runner no longer fit, and S3 is overkill for an MVP.

**Caveat the team should fix:** the entire marked-up proposal references `src/lib/...` (`src/lib/auth.ts`, `src/lib/prisma.ts`, `src/lib/recommendations/`), but the actual code is under `src/backend/...`. The proposal status notes are otherwise accurate, but every file path in them is stale.

#### d. One confusing thing

The **raw-SQL hydration step** in `recommendations/index.ts` (`hydrateExtendedRecommendationData`). The app already uses Prisma with typed models, yet a whole set of "newer" preference fields  are fetched through a hand-written raw SQL `JOIN` and then `Object.assign`-ed back onto the Prisma objects, guarded by `ensureExtendedRecommendationColumns`. It was confusing why structured columns that exist in `schema.prisma` are read through raw SQL instead of the normal Prisma `include`. I read `DESIGN.md`, `AGENTS.md`, and `recommendation-schema.ts` to understand it .the explanation is that the generated Prisma client was stale in their sandbox, so this is a deliberate temporary workaround. It works, but it means preference fields effectively live in two access paths, which is easy to get out of sync.

#### e. A conversation starter for Tuesday

I'd love to see them **demo two seeded users matching live and then justify the number** — e.g. "Person A scores 78% with Person B: walk us through how the both-directions scoring and the `min()` of the two sides produced that, which priority categories bumped the weights, and which penalty (cleanliness/sleep/etc.) got subtracted." That single trace would show whether the score and the displayed reasons/concerns actually agree.

### 2. Suggestions

#### a. Scope feedback for the final deliverable

Their three post-deliverable goals were: (1) recommendation robustness + user testing, maybe ML; (2) roommate-group feature; (3) housing-listing aggregation (already dropped). Given what's working, i recommend prioritize (1) robustness and real user testing**. The group feature (2) is net-new models and UI; reasonable to keep as a stretch goal, not the focus.

Two grounded finds:

- **Hard constraints can empty out the dashboard.** `filterHardConstraints` excludes on gender, smoking, pets, substances, alcohol, overnight guests, couples, *and* remote work. With a realistically small pool of UCSD test users, stacking that many one-strike exclusions will frequently produce the "No recommendations right now" empty state, which will read as "the app is broken" during user testing. I'd soften the non-safety ones before final.
- **Sensitive-data exposure on a live URL.** The only barrier to seeing other students' profiles (age, gender identity, budgets, lifestyle) is the `@ucsd.edu` gate — anyone with any UCSD email sees every `visible` profile. Combined with messaging that has **no blocking, reporting, or rate limiting**, putting this on a public Vercel URL with real students is a safety gap worth addressing before launch. Also, profile photos are stored as data URLs directly in Postgres (`User.image`), so every candidate row in the recommendation query drags a full image blob — that will bloat the DB and slow the feed as users grow.

#### b. One concrete suggestion

**Make the score penalties symmetric.** The overall score is computed as the *lower* of the two directional weighted scores (nice and mutual), but the penalties right after it are applied using only the current-user direction (`aToB`) — e.g. `if (aToB.cleanliness < 40) overallScore -= 12`, and the same for sleep, guests, relationship, and bathroom. Because the category scores returned to the UI use `min(aToB, bToA)` per category, but the penalties read raw `aToB`, **Person A and Person B can end up with different overall scores for the exact same pair**, and the penalty can fire (or not) inconsistently with the concern text shown. Switching the penalty checks to use the same `Math.min(aToB[cat], bToA[cat])` the UI already uses would make the score genuinely mutual and keep the number consistent with the "potential differences" list. It's a small change with a real correctness payoff for a tool whose whole pitch is a trustworthy, explainable match number.

#### c. Something you learned or thought was cool

The **conservative mutual scoring** design stuck with me. Instead of scoring "how much does A like B," it scores both directions and takes the min (`overallScore = Math.min(aToBScore, bToAScore)`), and even the per-category numbers shown in the UI are the lower of the two sides. That's a clean, opinionated answer to a real problem in matching apps, a match is only as strong as the less enthusiastic person and it pairs naturally with the priority-weighting step that lets each user re-weight categories and then renormalizes. I hadn't thought about applying a "weakest-link" rule to compatibility scoring before, and it's a genuinely smart, transparent alternative to throwing the data at a model.