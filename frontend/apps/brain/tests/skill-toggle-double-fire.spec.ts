import { test, expect } from '@playwright/test';

// Double-fire hypothesis — browser-verified against the real backend.
//
// SkillsView and CapabilitiesView faithfully port a NESTED <label> structure
// from the legacy code (skills.js:164-167/196-199, capabilities.js:97-100):
//
//   <label class="switch-label skill-toggle-wrap">   ← outer label
//     <label class="switch">                          ← inner label
//       <input type="checkbox" class="skill-toggle">  ← the control
//       <span class="switch-track"></span>            ← what the user clicks
//     </label>
//   </label>
//
// Nested labels are invalid HTML and have historically double-activated their
// control (one user click → two `change` events → the checkbox flips twice and
// nets back to its original state). The skill toggle auto-saves on `change`
// (toggleSkill → PUT /api/skills/:id/toggle), so a double-fire would surface as
// TWO PUTs and NO net state change. This test clicks the visible track exactly
// once and proves a single user click produces exactly one toggle + one flip.
//
// The proof is a request counter installed via page.on('request') BEFORE the
// click. Double-activation is SYNCHRONOUS — both `change` events fire within the
// single click dispatch, so both PUTs are dispatched at click time. The counter
// increments on request dispatch, and we then await the first PUT's RESPONSE;
// since a request event always precedes its response, a synchronous second PUT
// is already counted by the time the first response lands. So `puts` is final
// the instant we read it — no fixed settle delay needed, and the counter cannot
// miss an extra PUT the way a post-hoc waitForResponse could.
//
// (The McpView switches use a single, un-nested <label class="switch"> — also
// faithful to legacy mcp.js — so they are not at double-fire risk and need no
// separate probe.)

test.describe('Brain SPA — nested-label switch does not double-fire', () => {
  test('one click on a skill toggle = one PUT and one state flip', async ({ page }) => {
    let puts = 0;
    page.on('request', (req) => {
      if (req.method() === 'PUT' && /\/api\/skills\/[^/]+\/toggle$/.test(req.url())) puts++;
    });

    await page.goto('/brain/skills');
    const wrap = page.locator('.skill-toggle-wrap').first();
    await expect(wrap).toBeVisible();

    const input = wrap.locator('input.skill-toggle');
    const track = wrap.locator('.switch-track');
    const initial = await input.isChecked();

    // Single user click on the visible track.
    const firstResp = page.waitForResponse(
      (r) => /\/api\/skills\/[^/]+\/toggle$/.test(r.url()) && r.request().method() === 'PUT',
    );
    await track.click();
    await firstResp;

    // The control flipped once…
    await expect(input).toBeChecked({ checked: !initial });
    // …and exactly one PUT was dispatched (a synchronous double-fire would have
    // already pushed the counter to 2 by the time the first response landed).
    expect(puts).toBe(1);

    // Restore env state — and confirm the inverse click is also single-fire.
    const secondResp = page.waitForResponse(
      (r) => /\/api\/skills\/[^/]+\/toggle$/.test(r.url()) && r.request().method() === 'PUT',
    );
    await track.click();
    await secondResp;
    await expect(input).toBeChecked({ checked: initial });
    expect(puts).toBe(2);
  });
});
