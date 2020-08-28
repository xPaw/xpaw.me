/*
 * ztext.js v0.0.1
 * https://bennettfeely.com/ztext
 * Licensed MIT | (c) 2020 Bennett Feely
 */
if (
	CSS.supports("transform-style", "preserve-3d") &&
	!window.matchMedia("(prefers-reduced-motion: reduce)").matches
) {
	// Default values
	const z_default = {
		depth: "1rem",
		eventRotation: "60deg",
		layers: 10,
		perspective: "500px",
	};
	const tilts = [];

	// Get all elements with the [data-z] attribute
	document.querySelectorAll("[data-z]").forEach(zDraw);

	window.addEventListener(
		"mousemove",
		(e) => {
			const x_pct = (e.clientX / window.innerWidth - 0.5) * 2;
			const y_pct = (e.clientY / window.innerHeight - 0.5) * 2;

			tilts.forEach((tilt) => tilt(x_pct, y_pct));
		},
		{passive: true}
	);

	window.addEventListener(
		"touchmove",
		(e) => {
			const x_pct = (e.touches[0].clientX / window.innerWidth - 0.5) * 2;
			const y_pct = (e.touches[0].clientY / window.innerHeight - 0.5) * 2;

			tilts.forEach((tilt) => tilt(x_pct, y_pct));
		},
		{passive: true}
	);

	function zDraw(z) {
		const depth = z_default.depth;
		const depth_unit = depth.match(/[a-z]+/)[0];
		const depth_numeral = parseFloat(depth.replace(depth_unit, ""));

		const event_rotation = z_default.eventRotation;
		const event_rotation_unit = event_rotation.match(/[a-z]+/)[0];
		const event_rotation_numeral = parseFloat(
			event_rotation.replace(event_rotation_unit, "")
		);

		const layers = z_default.layers;
		const perspective = z_default.perspective;

		// Grab the text and replace it with a new structure
		const text = z.innerHTML;
		z.innerHTML = "";
		z.style.display = "inline-block";
		z.style.position = "relative";
		z.style.perspective = perspective;

		// Create a wrapper span that will hold all the layers
		const zText = document.createElement("span");
		zText.setAttribute("class", "z-text");
		zText.style.display = "inline-block";
		zText.style.transformStyle = "preserve-3d";

		// Create a layer for transforms from JS to be applied
		// CSS is stupid that transforms cannot be applied individually
		const zLayers = document.createElement("span");
		zLayers.setAttribute("class", "z-layers");
		zLayers.style.display = "inline-block";
		zLayers.style.transformStyle = "preserve-3d";

		zText.append(zLayers);

		for (let i = 0; i < layers; i++) {
			const pct = i / layers;

			// Create a layer
			const zLayer = document.createElement("span");
			zLayer.setAttribute("class", "z-layer");
			zLayer.innerHTML = text;
			zLayer.style.display = "inline-block";

			// Shift the layer on the z axis
			const zTranslation = -(pct * depth_numeral) + depth_numeral / 2;
			const transform = "translateZ(" + zTranslation + depth_unit + ")";
			zLayer.style.transform = transform;

			// Manipulate duplicate layers
			if (i >= 1) {
				// Overlay duplicate layers on top of each other
				zLayer.style.position = "absolute";
				zLayer.style.top = 0;
				zLayer.style.left = 0;

				// Hide duplicate layres from screen readers and user interation
				zLayer.setAttribute("aria-hidden", "true");
				zLayer.style.pointerEvents = "none";
			}

			// Add layer to wrapper span
			zLayers.append(zLayer);
		}

		// Finish adding everything to the original element
		z.append(zText);

		tilts.push((x_pct, y_pct) => {
			// Multiply pct rotation by eventRotation and eventDirection
			const x_tilt = x_pct * event_rotation_numeral;
			const y_tilt = -y_pct * event_rotation_numeral;

			// Add unit to transform value
			const unit = event_rotation_unit;

			// Rotate .z-layers as a function of x and y coordinates
			const transform = "rotateX(" + y_tilt + unit + ") rotateY(" + x_tilt + unit + ")";
			zLayers.style.transform = transform;
		});
	}
}
