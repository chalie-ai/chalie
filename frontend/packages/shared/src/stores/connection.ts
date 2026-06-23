import { defineStore } from 'pinia';

export const useConnectionStore = defineStore('connection', {
  state: () => ({ connected: false }),
  actions: {
    setConnected(v: boolean) {
      this.connected = v;
    },
  },
});
