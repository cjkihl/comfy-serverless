import type React from "react";
import { useState } from "react";

interface ImageData {
	src: string;
	alt: string;
	timestamp: Date;
}

// Export a hook for components to use the image display functionality
export const useImageGallery = () => {
	const [images, setImages] = useState<ImageData[]>([]);
	const [imageCount, setImageCount] = useState(0);

	const displayImage = (imageData: string) => {
		const newCount = imageCount + 1;
		setImageCount(newCount);

		let imageSrc: string;
		if (imageData.startsWith("data:")) {
			imageSrc = imageData;
		} else {
			// Try WEBP first, then PNG, then JPEG
			imageSrc = `data:image/webp;base64,${imageData}`;
		}

		const newImage: ImageData = {
			alt: `Generated Image ${newCount}`,
			src: imageSrc,
			timestamp: new Date(),
		};

		setImages((prev) => [...prev, newImage]);
	};

	const ImageGalleryComponent: React.FC<{ className?: string }> = ({
		className = "",
	}) => (
		<div className={className}>
			<h2 className="mb-4 text-xl font-semibold text-gray-900">
				Generated Images
			</h2>
			<div className="bg-gray-50 rounded-lg p-5 min-h-48 max-h-96 overflow-y-auto">
				{images.length === 0 ? (
					<p className="text-gray-500 text-center">No images generated yet</p>
				) : (
					<div className="grid gap-4" id="image-gallery">
						{images.map((image, index) => (
							<div
								className="inline-block m-2 p-3 bg-white rounded-lg shadow-sm"
								key={index}
							>
								<img
									alt={image.alt}
									className="max-w-sm rounded"
									onError={(e) => {
										const target = e.target as HTMLImageElement;
										if (target.src.includes("webp")) {
											target.src = target.src.replace("webp", "png");
										} else if (target.src.includes("png")) {
											target.src = target.src.replace("png", "jpeg");
										}
									}}
									src={image.src}
								/>
								<div className="text-sm text-gray-600 mt-2">
									{image.alt} - {image.timestamp.toLocaleTimeString()}
								</div>
							</div>
						))}
					</div>
				)}
			</div>
		</div>
	);

	return {
		displayImage,
		ImageGalleryComponent,
		imageCount,
		images,
	};
};
