import { render } from 'preact'
import { BrowserRouter } from 'react-router'
import { App } from './app.jsx'

render(
    <BrowserRouter>
        <App />
    </BrowserRouter>
, document.getElementById('app'))
