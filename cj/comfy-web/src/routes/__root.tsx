/// <reference types="vite/client" />

import {
	createRootRoute,
	HeadContent,
	Link,
	Outlet,
	Scripts,
} from "@tanstack/react-router";
import { TanStackRouterDevtools } from "@tanstack/router-devtools";
import type { ReactNode } from "react";
import "../global.css";

export const Route = createRootRoute({
	component: RootComponent,
	head: () => ({
		meta: [
			{
				charSet: "utf-8",
			},
			{
				content: "width=device-width, initial-scale=1",
				name: "viewport",
			},
			{
				title: "ComfyClient Test Suite",
			},
		],
	}),
});

function RootComponent() {
	return (
		<RootDocument>
			<div className="root-layout">
				<header className="navigation">
					<h1>🎨 ComfyClient Test Suite</h1>
					<nav>
						<Link
							activeProps={{ className: "active" }}
							className="nav-link"
							to="/"
						>
							Manual Testing
						</Link>

						<Link
							activeProps={{ className: "active" }}
							className="nav-link"
							to="/auto"
						>
							Auto Testing
						</Link>
					</nav>
				</header>
				<main>
					<Outlet />
				</main>
			</div>
			{import.meta.env.DEV && <TanStackRouterDevtools />}
			<style>{`
        .root-layout {
          min-height: 100vh;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .navigation {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 20px;
          margin-bottom: 0;
        }
        
        .navigation h1 {
          margin: 0 0 10px 0;
          font-size: 1.8rem;
        }
        
        .navigation nav {
          display: flex;
          gap: 20px;
        }
        
        .nav-link {
          color: white;
          text-decoration: none;
          padding: 8px 16px;
          border-radius: 6px;
          transition: all 0.2s;
          background: rgba(255, 255, 255, 0.1);
        }
        
        .nav-link:hover {
          background: rgba(255, 255, 255, 0.2);
        }
        
        .nav-link.active {
          background: rgba(255, 255, 255, 0.3);
          font-weight: 600;
        }
        
        main {
          flex: 1;
        }
        
        body {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }
        
        *, *::before, *::after {
          box-sizing: inherit;
        }
      `}</style>
		</RootDocument>
	);
}

function RootDocument({ children }: Readonly<{ children: ReactNode }>) {
	return (
		<html lang="en">
			<head>
				<HeadContent />
			</head>
			<body>
				{children}
				<Scripts />
			</body>
		</html>
	);
}
