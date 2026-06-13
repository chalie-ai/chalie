import js from '@eslint/js';
import vue from 'eslint-plugin-vue';
import vueTs from '@vue/eslint-config-typescript';
import eslintConfigPrettier from 'eslint-config-prettier';

export default [
  {
    ignores: [
      '**/dist/**',
      '**/node_modules/**',
      '**/playwright-report/**',
      '**/test-results/**',
      // Legacy hand-rolled JS files — not part of the Vue migration
      'brain/**',
      'interface/**',
      'login/**',
      'on-boarding/**',
      'shared/**',
    ],
  },
  js.configs.recommended,
  ...vue.configs['flat/recommended'],
  ...vueTs(),
  {
    rules: {
      'vue/multi-word-component-names': 'off',
    },
  },
  eslintConfigPrettier,
];
