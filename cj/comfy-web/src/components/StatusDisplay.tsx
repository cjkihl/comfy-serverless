import type { ConnectionState } from "@cj/comfy-client";
import type React from "react";

interface StatusDisplayProps {
	status: ConnectionState;
}

const getStatusClassName = (status: ConnectionState): string => {
	const baseClasses =
		"inline-block px-4 py-2 rounded-full font-semibold text-sm mt-3";
	switch (status) {
		case "connected":
			return `${baseClasses} bg-green-500 text-white`;
		case "connecting":
		case "reconnecting":
			return `${baseClasses} bg-yellow-500 text-white`;
		default:
			return `${baseClasses} bg-red-500 text-white`;
	}
};

export const StatusDisplay: React.FC<StatusDisplayProps> = ({ status }) => {
	return <div className={getStatusClassName(status)}>{status}</div>;
};
