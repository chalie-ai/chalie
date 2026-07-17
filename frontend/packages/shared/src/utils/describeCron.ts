/**
 * Human-readable label for a 5-field crontab schedule.
 *
 * Arguments mirror the backend columns `cron_minute`/`cron_hour`/`cron_dom`/
 * `cron_month`/`cron_dow` (minute, hour, day-of-month, month, day-of-week);
 * `*` means "every". The label covers the shapes a user actually creates in
 * conversation — every minute, every N minutes/hours, hourly at :MM, daily at
 * a time, a single named weekday, the Mon–Fri "weekday" set, one or more
 * days-of-month, and a single month. Anything outside that vocabulary
 * (comma-unions on the wrong field, both day-of-month AND day-of-week
 * restricted — the Vixie OR quirk — exotic steps or ranges) falls back to the
 * raw crontab expression, which is always truthful. This is the single source
 * of truth for schedule labels across both apps; there is no every-prefix
 * invariant any more (the engine accepts full crontab).
 */
export function describeCron(
  minute: string,
  hour: string,
  day: string,
  month: string,
  weekday: string,
): string {
  const min = minute.trim();
  const hr = hour.trim();
  const dom = day.trim();
  const mon = month.trim();
  const dow = weekday.trim();

  const time = describeTime(min, hr);
  const days = describeDays(dom, dow);
  const monthSuffix = describeMonth(mon);
  if (!time || !days || monthSuffix === null) {
    return `${min} ${hr} ${dom} ${mon} ${dow}`; // exotic → truthful raw crontab
  }
  return timePhrase(time, days) + monthSuffix;
}

const WEEKDAYS = [
  'Sunday',
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
] as const;
const MONTHS = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
] as const;

const pad2 = (n: number): string => String(n).padStart(2, '0');
const isStar = (f: string): boolean => f === '*';
const asInt = (f: string): number | null => (/^\d+$/.test(f) ? Number(f) : null);
const asStep = (f: string): number | null => {
  const m = /^\*\/(\d+)$/.exec(f);
  return m ? Number(m[1]) : null;
};

type Time =
  | { kind: 'everyMinute' }
  | { kind: 'everyNMinutes'; n: number }
  | { kind: 'everyNHours'; n: number }
  | { kind: 'hourly'; mm: number }
  | { kind: 'daily'; hh: number; mm: number };

/** Minute+hour → a time cadence, or null when it isn't a shape we name. */
function describeTime(minute: string, hour: string): Time | null {
  const mInt = asInt(minute);
  const hInt = asInt(hour);
  const mStep = asStep(minute);
  const hStep = asStep(hour);
  if (isStar(minute) && isStar(hour)) return { kind: 'everyMinute' };
  if (mStep !== null && mStep > 0 && isStar(hour)) return { kind: 'everyNMinutes', n: mStep };
  if (mInt === 0 && hStep !== null && hStep > 0) return { kind: 'everyNHours', n: hStep };
  if (mInt !== null && isStar(hour)) return { kind: 'hourly', mm: mInt };
  if (mInt !== null && hInt !== null) return { kind: 'daily', hh: hInt, mm: mInt };
  return null;
}

type Days =
  | { kind: 'all' }
  | { kind: 'weekdayName'; label: string }
  | { kind: 'weekdays' }
  | { kind: 'monthDays'; label: string };

/**
 * Day-of-month + day-of-week → a day scope. When BOTH are restricted the
 * schedule fires on the Vixie OR of the two — too ambiguous to phrase, so we
 * return null and let the caller show the raw crontab.
 */
function describeDays(dom: string, dow: string): Days | null {
  const domRestricted = !isStar(dom);
  const dowRestricted = !isStar(dow);
  if (domRestricted && dowRestricted) return null;
  if (!domRestricted && !dowRestricted) return { kind: 'all' };
  return dowRestricted ? describeDow(dow) : describeDom(dom);
}

function describeDow(dow: string): Days | null {
  const single = asInt(dow);
  if (single !== null && single >= 0 && single <= 7) {
    return { kind: 'weekdayName', label: WEEKDAYS[single % 7] }; // 7 folds to Sunday
  }
  if (dow === '1-5') return { kind: 'weekdays' };
  return null;
}

function describeDom(dom: string): Days | null {
  const nums: number[] = [];
  for (const part of dom.split(',')) {
    const n = asInt(part.trim());
    if (n === null || n < 1 || n > 31) return null;
    nums.push(n);
  }
  return { kind: 'monthDays', label: joinDayList(nums) };
}

function joinDayList(nums: number[]): string {
  if (nums.length === 1) return `day ${nums[0]}`;
  if (nums.length === 2) return `day ${nums[0]} and ${nums[1]}`;
  return `days ${nums.slice(0, -1).join(', ')} and ${nums[nums.length - 1]}`;
}

function describeMonth(month: string): string | null {
  if (isStar(month)) return '';
  const n = asInt(month);
  if (n !== null && n >= 1 && n <= 12) return ` in ${MONTHS[n - 1]}`;
  return null;
}

/** Suffix clause naming the day scope for non-daily cadences ("on Monday"). */
function onScope(days: Days): string {
  switch (days.kind) {
    case 'weekdayName':
      return ` on ${days.label}`;
    case 'weekdays':
      return ' on weekdays';
    case 'monthDays':
      return ` on ${days.label}`;
    default:
      return '';
  }
}

function timePhrase(time: Time, days: Days): string {
  switch (time.kind) {
    case 'everyMinute':
      return `Every minute${onScope(days)}`;
    case 'everyNMinutes':
      return `Every ${time.n} minutes${onScope(days)}`;
    case 'everyNHours':
      return `Every ${time.n} hours${onScope(days)}`;
    case 'hourly':
      return `Every hour at :${pad2(time.mm)}${onScope(days)}`;
    case 'daily':
      return dailyPhrase(time.hh, time.mm, days);
  }
}

function dailyPhrase(hh: number, mm: number, days: Days): string {
  const at = `${pad2(hh)}:${pad2(mm)}`;
  switch (days.kind) {
    case 'all':
      return `Every day at ${at}`;
    case 'weekdayName':
      return `Every ${days.label} at ${at}`;
    case 'weekdays':
      return `Every weekday at ${at}`;
    case 'monthDays':
      return `Monthly on ${days.label} at ${at}`;
  }
}
