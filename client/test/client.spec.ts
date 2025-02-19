import { test, expect } from "@playwright/test";

test("RLCore Home Visible", async ({ page }) => {
  await page.goto("http://127.0.0.1:8000/#/");
  await expect(page.getByRole("heading", { name: "Home" })).toBeVisible();
});

test("RLCore Home Button", async ({ page }) => {
  await page.goto("http://127.0.0.1:8000/#/");
  await expect(page.getByRole("button")).toHaveText("count is 0");
  await page.getByRole("button").click();
  await expect(page.getByRole("button")).toHaveText("count is 1");
});

test("RL Core About Health Button", async ({ page }) => {
  await page.goto("http://127.0.0.1:8000/#/about");
  await page.getByRole("button", { name: "Re-fetch Health Status" }).click();
  await expect(page.getByRole("main")).toContainText("success");
  await expect(page.getByRole("main")).toContainText(
    '{ "status": "OK", "time": ',
  );
});

test("Download default config", async ({ page }) => {
  await page.goto("http://127.0.0.1:8000/#/setup");
  await page.getByRole("link", { name: "Next" }).click();
  await page.getByRole("link", { name: "Next" }).click();
  await page.getByRole("link", { name: "Next" }).click();
  const downloadPromise = page.waitForEvent("download");

  await page
    .getByRole("button", { name: "Generate Configuration YAML" })
    .click();
  await downloadPromise;
});
