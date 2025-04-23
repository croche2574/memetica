import { h, render } from 'preact';
import { App } from './src/App';

const root = document.getElementById('app');
if (root) render(<App />, root);