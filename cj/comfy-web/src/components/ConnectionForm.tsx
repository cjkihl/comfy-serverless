import type React from "react";
import type { ConnectionConfig } from "../hooks/useComfyClient";

interface ConnectionFormProps {
	config: ConnectionConfig;
	onConfigChange: (config: ConnectionConfig) => void;
	onConnect: () => void;
	onDisconnect: () => void;
	isConnected: boolean;
	isConnecting: boolean;
	autoConnect?: boolean;
}

export const ConnectionForm: React.FC<ConnectionFormProps> = ({
	config,
	onConfigChange,
	onConnect,
	onDisconnect,
	isConnected,
	isConnecting,
	autoConnect = false,
}) => {
	const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		onConfigChange({ ...config, url: e.currentTarget.value });
	};

	const handleTokenChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		onConfigChange({ ...config, jwtToken: e.currentTarget.value });
	};

	return (
		<div className="mb-8">
			<h2 className="mb-4 text-xl font-semibold text-gray-900">
				Connection Settings
			</h2>

			{autoConnect && (
				<div className="mb-4 rounded-lg bg-blue-50 border border-blue-200 p-4 text-blue-800">
					🔄 <strong>AutoConnect Mode:</strong> Connection will be established
					automatically and reconnect on tab focus.
				</div>
			)}

			<div className="mb-4">
				<label
					className="block mb-2 text-sm font-medium text-gray-700"
					htmlFor="wsUrl"
				>
					WebSocket URL:
				</label>
				<input
					className="w-full px-3 py-2 border-2 border-gray-200 rounded-lg text-sm transition-all focus:outline-none focus:border-indigo-500 focus:ring-3 focus:ring-indigo-100 disabled:opacity-50 disabled:cursor-not-allowed"
					disabled={isConnected || isConnecting}
					id="wsUrl"
					onChange={handleUrlChange}
					placeholder="ws://localhost:8190/ws"
					type="text"
					value={config.url}
				/>
			</div>

			<div className="mb-4">
				<label
					className="block mb-2 text-sm font-medium text-gray-700"
					htmlFor="jwtToken"
				>
					JWT Token (optional):
				</label>
				<input
					className="w-full px-3 py-2 border-2 border-gray-200 rounded-lg text-sm transition-all focus:outline-none focus:border-indigo-500 focus:ring-3 focus:ring-indigo-100 disabled:opacity-50 disabled:cursor-not-allowed"
					disabled={isConnected || isConnecting}
					id="jwtToken"
					onChange={handleTokenChange}
					placeholder="your-jwt-token"
					type="text"
					value={config.jwtToken || ""}
				/>
			</div>

			{!autoConnect && (
				<div className="flex gap-3 flex-wrap">
					<button
						className="px-6 py-3 bg-indigo-500 text-white font-semibold rounded-lg transition-all hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed"
						disabled={isConnected || isConnecting}
						onClick={onConnect}
					>
						{isConnecting ? "Connecting..." : "Connect"}
					</button>
					<button
						className="px-6 py-3 bg-red-500 text-white font-semibold rounded-lg transition-all hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed"
						disabled={!isConnected}
						onClick={onDisconnect}
					>
						Disconnect
					</button>
				</div>
			)}
		</div>
	);
};
