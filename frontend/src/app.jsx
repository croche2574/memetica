import { lazy, LocationProvider, ErrorBoundary, Router } from 'preact-iso';

import { Login } from './routes/login';
import { Search } from './routes/search';
import { NotFound } from './routes/notfound';

export const App = () => {

    return (
        <LocationProvider>
            <ErrorBoundary>
                <Router>
                    <Login path='/login'/>
                    <Search path='/search'/>
                    <NotFound default />
                </Router>
            </ErrorBoundary>
        </LocationProvider>
    )
}