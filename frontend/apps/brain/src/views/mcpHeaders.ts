/**
 * Header-row editing for the MCP outbound-server forms (the "Additional Headers"
 * key/value editor). Pure list operations, framework-agnostic, so they live
 * outside the SFC — which then owns only view state and API calls.
 */

/** One editable key/value pair in an MCP server's Additional Headers list. */
export interface HeaderRow {
  key: string;
  value: string;
}

/** Append a blank editable row. */
export function addHeaderRow(list: HeaderRow[]): void {
  list.push({ key: '', value: '' });
}

/** Remove the row at `idx`. */
export function removeHeaderRow(list: HeaderRow[], idx: number): void {
  list.splice(idx, 1);
}

/** Collapse the editable rows into a headers map, dropping blank-keyed rows. */
export function collectHeaders(list: HeaderRow[]): Record<string, string> {
  const headers: Record<string, string> = {};
  for (const row of list) {
    const k = row.key.trim();
    const v = row.value.trim();
    if (k) headers[k] = v;
  }
  return headers;
}
