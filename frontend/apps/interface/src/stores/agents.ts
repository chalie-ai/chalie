/**
 * Agents store — every subagent still running or waiting for its caller's
 * reply, keyed by its `agent_id`: what the Activity drawer lists under
 * "Agents", and part of what shows the top bar's Activity button.
 *
 * Read in full over REST on every WS connect (`refresh()`), first load and
 * reconnect alike, then kept current by the `agent_changed` frame the server
 * sends each time a row is written (`apply()`): a running or waiting agent is
 * added or updated, one that ended is removed. A frame that lands while a read
 * is in flight wins over that read, so a read sent before an agent ended never
 * brings it back; only the newest read applies.
 */
import { defineStore } from 'pinia';
import { agents as agentsApi } from '../api';
import type { Agent } from '../api/agents';

/** Bumped by every read; a read whose number is stale is dropped. */
let refreshSeq = 0;

/** The frames applied since the newest read began, by agent — laid over that
 *  read when it lands. Null while no read is in flight. */
let framedSinceRead: Map<number, Agent> | null = null;

/** Put `agent` in `byId` while it runs or waits for a reply; take it out
 *  once it ended. */
function place(byId: Map<number, Agent>, agent: Agent): void {
  if (agent.status === 'running' || agent.status === 'waiting') byId.set(agent.agent_id, agent);
  else byId.delete(agent.agent_id);
}

export const useAgentsStore = defineStore('agents', {
  state: () => ({
    byId: new Map<number, Agent>(),
  }),

  getters: {
    /** The running and waiting agents, oldest first. */
    active(state): Agent[] {
      return [...state.byId.values()].sort((a, b) => a.agent_id - b.agent_id);
    },
  },

  actions: {
    /** One `agent_changed` frame: the row as it was just written. */
    apply(agent: Agent): void {
      framedSinceRead?.set(agent.agent_id, agent);
      place(this.byId, agent);
    },

    /**
     * Re-read every running and waiting agent. Best-effort: a failed read
     * leaves the list as it was; the next connect retries.
     */
    async refresh(): Promise<void> {
      const seq = ++refreshSeq;
      const framed = new Map<number, Agent>();
      framedSinceRead = framed;
      let rows: Agent[];
      try {
        rows = await agentsApi.active();
      } catch (err) {
        console.warn('[agents] active agents fetch failed:', err);
        if (seq === refreshSeq) framedSinceRead = null;
        return;
      }
      if (seq !== refreshSeq) return;
      framedSinceRead = null;
      const byId = new Map(rows.map((agent) => [agent.agent_id, agent]));
      for (const agent of framed.values()) place(byId, agent);
      this.byId = byId;
    },
  },
});
