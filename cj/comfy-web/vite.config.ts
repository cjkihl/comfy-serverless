// @ts-nocheck
import tailwindcss from "@tailwindcss/vite";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import tsConfigPaths from "vite-tsconfig-paths";

export default defineConfig({
	plugins: [
		tsConfigPaths(),
		tanstackStart(),
		// react's vite plugin must come after start's vite plugin
		react(),
		tailwindcss(),
	],
	server: {
		host: "127.0.0.1",
		port: 5173,
	},
});
