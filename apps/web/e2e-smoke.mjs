import { chromium } from 'playwright'

const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
const errors = []
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()) })
page.on('pageerror', error => errors.push(error.message))
await page.goto('http://127.0.0.1:5173', { waitUntil: 'networkidle' })
await page.locator('input[type=file]').setInputFiles('D:/shayo/projects/footballanalysis/runs/football-frame-test.mp4')
await page.waitForTimeout(3000)
await page.getByText('Ready for detection').waitFor({ timeout: 15000 })
await page.getByRole('button', { name: 'Run detector' }).click()
await page.getByText('Computer vision report').waitFor({ timeout: 60000 })
const reportVisible = await page.getByText('Stable track segments').isVisible()
await page.screenshot({ path: 'D:/shayo/projects/footballanalysis/runs/frontend-analysis-e2e.png', fullPage: true })
await page.getByRole('button', { name: 'Video', exact: true }).click()
const videoVisible = await page.locator('video').isVisible()
await page.getByRole('button', { name: 'Calibrate pitch' }).click()
const calibration = page.getByRole('button', { name: /Mark pitch corner/ })
const bounds = await calibration.boundingBox()
if (!bounds) throw new Error('Calibration overlay has no bounds')
for (const [x, y] of [[.12, .15], [.88, .15], [.88, .88], [.12, .88]]) {
  await calibration.click({ position: { x: bounds.width * x, y: bounds.height * y } })
}
await page.getByRole('button', { name: 'Tactical pitch' }).click()
const pitchPlayers = await page.locator('.player-marker').count()
await page.screenshot({ path: 'D:/shayo/projects/footballanalysis/runs/frontend-e2e.png', fullPage: true })
console.log(JSON.stringify({ reportVisible, videoVisible, pitchPlayers, errors }))
await browser.close()
