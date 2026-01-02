import type React from "react";
import { useEffect, useRef, useState } from "react";
import type { LogEntry, UILogger } from "../shared/logger";

interface LogViewerProps {
	logger: UILogger;
}

export const LogViewer: React.FC<LogViewerProps> = ({ logger }) => {
	const [entries, setEntries] = useState<LogEntry[]>(logger.getEntries());
	const logRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		const unsubscribe = logger.subscribe(setEntries);
		return unsubscribe;
	}, [logger]);

	useEffect(() => {
		// Auto-scroll to bottom when new entries are added
		if (logRef.current) {
			logRef.current.scrollTop = logRef.current.scrollHeight;
		}
	});

	const handleClear = () => {
		logger.clear();
	};

	const getLogClassName = (type: LogEntry["type"]): string => {
		switch (type) {
			case "error":
				return "text-red-400";
			case "success":
				return "text-green-400";
			case "warn":
				return "text-yellow-400";
			case "debug":
				return "text-gray-400";
			default:
				return "text-blue-400";
		}
	};

	return (
		<div className="mb-8" id="log-viewer">
			<h2 className="mb-4 text-xl font-semibold text-gray-900">Event Log</h2>
			<button
				className="mb-3 px-4 py-2 bg-gray-600 text-white text-sm font-semibold rounded-lg transition-all hover:bg-gray-700"
				onClick={handleClear}
			>
				Clear Log
			</button>
			<div
				className="bg-gray-900 text-gray-100 rounded-lg p-5 h-96 overflow-y-auto font-mono text-sm leading-relaxed wrap-break-word"
				ref={logRef}
			>
				{entries.map((entry, index) => (
					<div
						className="mb-2 p-2 rounded bg-gray-800/50 wrap-break-word whitespace-pre-wrap"
						key={index}
					>
						<span className="text-gray-400">[{entry.timestamp}]</span>{" "}
						<span className={getLogClassName(entry.type)}>{entry.message}</span>
					</div>
				))}
			</div>
		</div>
	);
};
