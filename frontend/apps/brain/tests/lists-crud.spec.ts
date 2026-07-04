import { expect, test } from '@playwright/test';

// A real CRUD round-trip against the live backend: create a list via the modal
// (POST /api/lists), assert the new card renders, then delete it via the card's
// Delete button + the confirm dialog (DELETE /api/lists/:id) and assert the card
// is gone. No mocks — the list genuinely persists and is genuinely removed, so a
// break anywhere in the request → render → delete chain fails loudly. The test
// cleans up after itself (the list it creates is the list it deletes).

test.describe('Brain SPA — Lists CRUD round-trip', () => {
  test('create a list, see it, delete it', async ({ page }) => {
    const name = `pw-e2e-${Date.now()}`;

    await page.goto('/brain/lists');
    await expect(page.getByRole('heading', { name: 'Lists', exact: true })).toBeVisible();

    // Open the create modal and submit.
    await page.locator('.panel-header').getByRole('button', { name: 'New List' }).click();
    const nameInput = page.locator('#newListName');
    await expect(nameInput).toBeVisible();
    await nameInput.fill(name);
    await page.locator('.modal').getByRole('button', { name: 'Create' }).click();

    // The new card renders with the typed name (real POST round-trip).
    const card = page.locator('.list-card', { has: page.getByText(name, { exact: true }) });
    await expect(card).toBeVisible();

    // Delete it → confirm dialog → DELETE round-trip → card disappears.
    await card.getByRole('button', { name: 'Delete' }).click();
    await page.locator('.modal-actions button.btn-danger').click();
    await expect(card).toHaveCount(0);
  });
});
