// Pretty much copied David's wave simulation which is licensed under MIT <3
// Taken from https://github.com/dli/waves

(function(){

var INITIAL_SIZE = 2048,
	INITIAL_WIND = [0.0, 100.0],
	INITIAL_CHOPPINESS = 4;

var CLEAR_COLOR = [0.0, 0.0, 0.0, 1.0],
	GEOMETRY_ORIGIN = [-1000.0, -1400.0],
	SUN_DIRECTION = [0.1, 0.1, 0.0],
	OCEAN_COLOR = [0.004, 0.016, 0.047],
	SKY_COLOR = [3.2, 9.6, 12.8],
	EXPOSURE = 0.5,
	GEOMETRY_RESOLUTION = 256,
	GEOMETRY_SIZE = 2048,
	RESOLUTION = 512;

var SIZE_OF_FLOAT = 4;

var OCEAN_COORDINATES_UNIT = 1;

var INITIAL_SPECTRUM_UNIT = 0,
	SPECTRUM_UNIT = 1,
	DISPLACEMENT_MAP_UNIT = 2,
	NORMAL_MAP_UNIT = 3,
	PING_PHASE_UNIT = 4,
	PONG_PHASE_UNIT = 5,
	PING_TRANSFORM_UNIT = 6,
	PONG_TRANSFORM_UNIT = 7;

var FOV = (50 / 180) * Math.PI,
	NEAR = 1,
	FAR = 5000,
	MIN_ASPECT = 16 / 9;

var SIMULATOR_CANVAS_ID = 'simulator';

var CAMERA_DISTANCE = 512,
	INITIAL_ELEVATION = 0.9;

var makeIdentityMatrix = function (matrix) {
	matrix[0] = 1.0;
	matrix[1] = 0.0;
	matrix[2] = 0.0;
	matrix[3] = 0.0;
	matrix[4] = 0.0;
	matrix[5] = 1.0;
	matrix[6] = 0.0;
	matrix[7] = 0.0;
	matrix[8] = 0.0;
	matrix[9] = 0.0;
	matrix[10] = 1.0;
	matrix[11] = 0.0;
	matrix[12] = 0.0;
	matrix[13] = 0.0;
	matrix[14] = 0.0;
	matrix[15] = 1.0;
	return matrix;
};

var makeXRotationMatrix = function (matrix, angle) {
	matrix[0] = 1.0;
	matrix[1] = 0.0;
	matrix[2] = 0.0;
	matrix[3] = 0.0;
	matrix[4] = 0.0;
	matrix[5] = Math.cos(angle);
	matrix[6] = Math.sin(angle);
	matrix[7] = 0.0;
	matrix[8] = 0.0;
	matrix[9] = -Math.sin(angle);
	matrix[10] = Math.cos(angle);
	matrix[11] = 0.0;
	matrix[12] = 0.0;
	matrix[13] = 0.0;
	matrix[14] = 0.0;
	matrix[15] = 1.0;
	return matrix;
};

var premultiplyMatrix = function (out, matrixA, matrixB) {
	var b0 = matrixB[0], b4 = matrixB[4], b8 = matrixB[8], b12 = matrixB[12],
		b1 = matrixB[1], b5 = matrixB[5], b9 = matrixB[9], b13 = matrixB[13],
		b2 = matrixB[2], b6 = matrixB[6], b10 = matrixB[10], b14 = matrixB[14],
		b3 = matrixB[3], b7 = matrixB[7], b11 = matrixB[11], b15 = matrixB[15],

		aX = matrixA[0], aY = matrixA[1], aZ = matrixA[2], aW = matrixA[3];
	out[0] = b0 * aX + b4 * aY + b8 * aZ + b12 * aW;
	out[1] = b1 * aX + b5 * aY + b9 * aZ + b13 * aW;
	out[2] = b2 * aX + b6 * aY + b10 * aZ + b14 * aW;
	out[3] = b3 * aX + b7 * aY + b11 * aZ + b15 * aW;

	aX = matrixA[4], aY = matrixA[5], aZ = matrixA[6], aW = matrixA[7];
	out[4] = b0 * aX + b4 * aY + b8 * aZ + b12 * aW;
	out[5] = b1 * aX + b5 * aY + b9 * aZ + b13 * aW;
	out[6] = b2 * aX + b6 * aY + b10 * aZ + b14 * aW;
	out[7] = b3 * aX + b7 * aY + b11 * aZ + b15 * aW;

	aX = matrixA[8], aY = matrixA[9], aZ = matrixA[10], aW = matrixA[11];
	out[8] = b0 * aX + b4 * aY + b8 * aZ + b12 * aW;
	out[9] = b1 * aX + b5 * aY + b9 * aZ + b13 * aW;
	out[10] = b2 * aX + b6 * aY + b10 * aZ + b14 * aW;
	out[11] = b3 * aX + b7 * aY + b11 * aZ + b15 * aW;

	aX = matrixA[12], aY = matrixA[13], aZ = matrixA[14], aW = matrixA[15];
	out[12] = b0 * aX + b4 * aY + b8 * aZ + b12 * aW;
	out[13] = b1 * aX + b5 * aY + b9 * aZ + b13 * aW;
	out[14] = b2 * aX + b6 * aY + b10 * aZ + b14 * aW;
	out[15] = b3 * aX + b7 * aY + b11 * aZ + b15 * aW;

	return out;
};

var makePerspectiveMatrix = function (matrix, fov, aspect, near, far) {
	var f = Math.tan(0.5 * (Math.PI - fov)),
		range = near - far;

	matrix[0] = f / aspect;
	matrix[1] = 0;
	matrix[2] = 0;
	matrix[3] = 0;
	matrix[4] = 0;
	matrix[5] = f;
	matrix[6] = 0;
	matrix[7] = 0;
	matrix[8] = 0;
	matrix[9] = 0;
	matrix[10] = far / range;
	matrix[11] = -1;
	matrix[12] = 0;
	matrix[13] = 0;
	matrix[14] = (near * far) / range;
	matrix[15] = 0.0;

	return matrix;
};

var log2 = function (number) {
	return Math.log(number) / Math.log(2);
};

var buildProgramWrapper = function (gl, vertexShader, fragmentShader, attributeLocations) {
	var programWrapper = {};

	var program = gl.createProgram();
	gl.attachShader(program, vertexShader);
	gl.attachShader(program, fragmentShader);
	for (var attributeName in attributeLocations) {
		gl.bindAttribLocation(program, attributeLocations[attributeName], attributeName);
	}
	gl.linkProgram(program);
	var uniformLocations = {};
	var numberOfUniforms = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
	for (var i = 0; i < numberOfUniforms; i += 1) {
		var activeUniform = gl.getActiveUniform(program, i),
			uniformLocation = gl.getUniformLocation(program, activeUniform.name);
		uniformLocations[activeUniform.name] = uniformLocation;
	}

	programWrapper.program = program;
	programWrapper.uniformLocations = uniformLocations;

	return programWrapper;
};

var buildShader = function (gl, type, source) {
	var shader = gl.createShader(type);
	gl.shaderSource(shader, source);
	gl.compileShader(shader);
	return shader;
};

var buildTexture = function (gl, unit, format, type, width, height, data, wrapS, wrapT, minFilter, magFilter) {
	var texture = gl.createTexture();
	gl.activeTexture(gl.TEXTURE0 + unit);
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.texImage2D(gl.TEXTURE_2D, 0, format, width, height, 0, format, type, data);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrapS);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrapT);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, minFilter);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, magFilter);
	return texture;
};

var buildFramebuffer = function (gl, attachment) {
	var framebuffer = gl.createFramebuffer();
	gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
	gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, attachment, 0);
	return framebuffer;
};

var hasWebGLSupportWithExtensions = function (extensions) {
	var canvas = document.createElement('canvas');
	var gl = null;
	try {
		gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
	} catch (e) {
		return false;
	}
	if (gl === null) {
		return false;
	}

	for (var i = 0; i < extensions.length; ++i) {
		if (gl.getExtension(extensions[i]) === null) {
			return false;
		}
	}

	return true;
};

var Camera = function () {
	var elevation = INITIAL_ELEVATION,
		viewMatrix = makeIdentityMatrix(new Float32Array(16)),
		position = new Float32Array(3);

	this.getPosition = function () {
		return position;
	};

	var orbitTranslationMatrix = makeIdentityMatrix(new Float32Array(16)),
		xRotationMatrix = new Float32Array(16),
		distanceTranslationMatrix = makeIdentityMatrix(new Float32Array(16));

	this.getViewMatrix = function () {
		makeIdentityMatrix(viewMatrix);

		makeXRotationMatrix(xRotationMatrix, elevation);
		
		distanceTranslationMatrix[14] = -CAMERA_DISTANCE;

		premultiplyMatrix(viewMatrix, viewMatrix, orbitTranslationMatrix);
		premultiplyMatrix(viewMatrix, viewMatrix, xRotationMatrix);
		premultiplyMatrix(viewMatrix, viewMatrix, distanceTranslationMatrix);

		position[0] = 0;
		position[1] = CAMERA_DISTANCE * Math.cos(Math.PI / 2 - elevation);
		position[2] = CAMERA_DISTANCE * Math.sin(Math.PI / 2 - elevation);

		return viewMatrix;
	};
};

var FULLSCREEN_VERTEX_SOURCE = [
	'attribute vec2 a_position;',
	'varying vec2 v_coordinates;',

	'void main (void) {',
		'v_coordinates = a_position * 0.5 + 0.5;',
		'gl_Position = vec4(a_position, 0.0, 1.0);',
	'}',
].join('\n');

//GPU FFT using the Stockham formulation
var SUBTRANSFORM_FRAGMENT_SOURCE = [
	'precision highp float;',

	'const float PI = 3.14159265359;',

	'uniform sampler2D u_input;',

	'uniform float u_transformSize;',
	'uniform float u_subtransformSize;',

	'varying vec2 v_coordinates;',

	'vec2 multiplyComplex (vec2 a, vec2 b) {',
		'return vec2(a[0] * b[0] - a[1] * b[1], a[1] * b[0] + a[0] * b[1]);',
	'}',

	'void main (void) {',

		'#ifdef HORIZONTAL',
		'float index = v_coordinates.x * u_transformSize - 0.5;',
		'#else',
		'float index = v_coordinates.y * u_transformSize - 0.5;',
		'#endif',

		'float evenIndex = floor(index / u_subtransformSize) * (u_subtransformSize * 0.5) + mod(index, u_subtransformSize * 0.5);',
		
		//transform two complex sequences simultaneously
		'#ifdef HORIZONTAL',
		'vec4 even = texture2D(u_input, vec2(evenIndex + 0.5, gl_FragCoord.y) / u_transformSize).rgba;',
		'vec4 odd = texture2D(u_input, vec2(evenIndex + u_transformSize * 0.5 + 0.5, gl_FragCoord.y) / u_transformSize).rgba;',
		'#else',
		'vec4 even = texture2D(u_input, vec2(gl_FragCoord.x, evenIndex + 0.5) / u_transformSize).rgba;',
		'vec4 odd = texture2D(u_input, vec2(gl_FragCoord.x, evenIndex + u_transformSize * 0.5 + 0.5) / u_transformSize).rgba;',
		'#endif',

		'float twiddleArgument = -2.0 * PI * (index / u_subtransformSize);',
		'vec2 twiddle = vec2(cos(twiddleArgument), sin(twiddleArgument));',

		'vec2 outputA = even.xy + multiplyComplex(twiddle, odd.xy);',
		'vec2 outputB = even.zw + multiplyComplex(twiddle, odd.zw);',

		'gl_FragColor = vec4(outputA, outputB);',
	'}'
].join('\n');

var INITIAL_SPECTRUM_FRAGMENT_SOURCE = [
	'precision highp float;',

	'const float PI = 3.14159265359;',
	'const float G = 9.81;',
	'const float KM = 370.0;',
	'const float CM = 0.23;',

	'uniform vec2 u_wind;',
	'uniform float u_resolution;',
	'uniform float u_size;',

	'float square (float x) {',
		'return x * x;',
	'}',

	'float omega (float k) {',
		'return sqrt(G * k * (1.0 + square(k / KM)));',
	'}',

	'float tanh (float x) {',
		'return (1.0 - exp(-2.0 * x)) / (1.0 + exp(-2.0 * x));',
	'}',

	'void main (void) {',
		'vec2 coordinates = gl_FragCoord.xy - 0.5;',
		'float n = (coordinates.x < u_resolution * 0.5) ? coordinates.x : coordinates.x - u_resolution;',
		'float m = (coordinates.y < u_resolution * 0.5) ? coordinates.y : coordinates.y - u_resolution;',
		'vec2 waveVector = (2.0 * PI * vec2(n, m)) / u_size;',
		'float k = length(waveVector);',

		'float U10 = length(u_wind);',

		'float Omega = 0.84;',
		'float kp = G * square(Omega / U10);',

		'float c = omega(k) / k;',
		'float cp = omega(kp) / kp;',

		'float Lpm = exp(-1.25 * square(kp / k));',
		'float gamma = 1.7;',
		'float sigma = 0.08 * (1.0 + 4.0 * pow(Omega, -3.0));',
		'float Gamma = exp(-square(sqrt(k / kp) - 1.0) / 2.0 * square(sigma));',
		'float Jp = pow(gamma, Gamma);',
		'float Fp = Lpm * Jp * exp(-Omega / sqrt(10.0) * (sqrt(k / kp) - 1.0));',
		'float alphap = 0.006 * sqrt(Omega);',
		'float Bl = 0.5 * alphap * cp / c * Fp;',

		'float z0 = 0.000037 * square(U10) / G * pow(U10 / cp, 0.9);',
		'float uStar = 0.41 * U10 / log(10.0 / z0);',
		'float alpham = 0.01 * ((uStar < CM) ? (1.0 + log(uStar / CM)) : (1.0 + 3.0 * log(uStar / CM)));',
		'float Fm = exp(-0.25 * square(k / KM - 1.0));',
		'float Bh = 0.5 * alpham * CM / c * Fm * Lpm;',

		'float a0 = log(2.0) / 4.0;',
		'float am = 0.13 * uStar / CM;',
		'float Delta = tanh(a0 + 4.0 * pow(c / cp, 2.5) + am * pow(CM / c, 2.5));',

		'float cosPhi = dot(normalize(u_wind), normalize(waveVector));',

		'float S = (1.0 / (2.0 * PI)) * pow(k, -4.0) * (Bl + Bh) * (1.0 + Delta * (2.0 * cosPhi * cosPhi - 1.0));',

		'float dk = 2.0 * PI / u_size;',
		'float h = sqrt(S / 2.0) * dk;',

		'if (waveVector.x == 0.0 && waveVector.y == 0.0) {',
			'h = 0.0;', //no DC term
		'}',

		'gl_FragColor = vec4(h, 0.0, 0.0, 0.0);',
	'}'
].join('\n');

//phases stored in separate texture to ensure wave continuity on resizing
var PHASE_FRAGMENT_SOURCE = [
	'precision highp float;',

	'const float PI = 3.14159265359;',
	'const float G = 9.81;',
	'const float KM = 370.0;',

	'varying vec2 v_coordinates;',

	'uniform sampler2D u_phases;',

	'uniform float u_deltaTime;',
	'uniform float u_resolution;',
	'uniform float u_size;',

	'float omega (float k) {',
		'return sqrt(G * k * (1.0 + k * k / KM * KM));',
	'}',

	'void main (void) {',
		'float deltaTime = 1.0 / 60.0;',
		'vec2 coordinates = gl_FragCoord.xy - 0.5;',
		'float n = (coordinates.x < u_resolution * 0.5) ? coordinates.x : coordinates.x - u_resolution;',
		'float m = (coordinates.y < u_resolution * 0.5) ? coordinates.y : coordinates.y - u_resolution;',
		'vec2 waveVector = (2.0 * PI * vec2(n, m)) / u_size;',

		'float phase = texture2D(u_phases, v_coordinates).r;',
		'float deltaPhase = omega(length(waveVector)) * (u_deltaTime / 1000.0);',
		'phase = mod(phase + deltaPhase, 2.0 * PI);',

		'gl_FragColor = vec4(phase, 0.0, 0.0, 0.0);',
	'}'
].join('\n');

var SPECTRUM_FRAGMENT_SOURCE = [
	'precision highp float;',

	'const float PI = 3.14159265359;',
	'const float G = 9.81;',
	'const float KM = 370.0;',

	'varying vec2 v_coordinates;',

	'uniform float u_size;',
	'uniform float u_resolution;',

	'uniform sampler2D u_phases;',
	'uniform sampler2D u_initialSpectrum;',

	'uniform float u_choppiness;',

	'vec2 multiplyComplex (vec2 a, vec2 b) {',
		'return vec2(a[0] * b[0] - a[1] * b[1], a[1] * b[0] + a[0] * b[1]);',
	'}',

	'vec2 multiplyByI (vec2 z) {',
		'return vec2(-z[1], z[0]);',
	'}',

	'float omega (float k) {',
		'return sqrt(G * k * (1.0 + k * k / KM * KM));',
	'}',

	'void main (void) {',
		'vec2 coordinates = gl_FragCoord.xy - 0.5;',
		'float n = (coordinates.x < u_resolution * 0.5) ? coordinates.x : coordinates.x - u_resolution;',
		'float m = (coordinates.y < u_resolution * 0.5) ? coordinates.y : coordinates.y - u_resolution;',
		'vec2 waveVector = (2.0 * PI * vec2(n, m)) / u_size;',

		'float phase = texture2D(u_phases, v_coordinates).r;',
		'vec2 phaseVector = vec2(cos(phase), sin(phase));',

		'vec2 h0 = texture2D(u_initialSpectrum, v_coordinates).rg;',
		'vec2 h0Star = texture2D(u_initialSpectrum, vec2(1.0 - v_coordinates + 1.0 / u_resolution)).rg;',
		'h0Star.y *= -1.0;',

		'vec2 h = multiplyComplex(h0, phaseVector) + multiplyComplex(h0Star, vec2(phaseVector.x, -phaseVector.y));',

		'vec2 hX = -multiplyByI(h * (waveVector.x / length(waveVector))) * u_choppiness;',
		'vec2 hZ = -multiplyByI(h * (waveVector.y / length(waveVector))) * u_choppiness;',

		//no DC term
		'if (waveVector.x == 0.0 && waveVector.y == 0.0) {',
			'h = vec2(0.0);',
			'hX = vec2(0.0);',
			'hZ = vec2(0.0);',
		'}',

		'gl_FragColor = vec4(hX + multiplyByI(h), hZ);',
	'}'
].join('\n');

//cannot use common heightmap optimisations because displacements are horizontal as well as vertical
var NORMAL_MAP_FRAGMENT_SOURCE = [
	'precision highp float;',

	'varying vec2 v_coordinates;',

	'uniform sampler2D u_displacementMap;',
	'uniform float u_resolution;',
	'uniform float u_size;',

	'void main (void) {',
		'float texel = 1.0 / u_resolution;',
		'float texelSize = u_size / u_resolution;',

		'vec3 center = texture2D(u_displacementMap, v_coordinates).rgb;',
		'vec3 right = vec3(texelSize, 0.0, 0.0) + texture2D(u_displacementMap, v_coordinates + vec2(texel, 0.0)).rgb - center;',
		'vec3 left = vec3(-texelSize, 0.0, 0.0) + texture2D(u_displacementMap, v_coordinates + vec2(-texel, 0.0)).rgb - center;',
		'vec3 top = vec3(0.0, 0.0, -texelSize) + texture2D(u_displacementMap, v_coordinates + vec2(0.0, -texel)).rgb - center;',
		'vec3 bottom = vec3(0.0, 0.0, texelSize) + texture2D(u_displacementMap, v_coordinates + vec2(0.0, texel)).rgb - center;',

		'vec3 topRight = cross(right, top);',
		'vec3 topLeft = cross(top, left);',
		'vec3 bottomLeft = cross(left, bottom);',
		'vec3 bottomRight = cross(bottom, right);',

		'gl_FragColor = vec4(normalize(topRight + topLeft + bottomLeft + bottomRight), 1.0);',
	'}'
].join('\n');

var OCEAN_VERTEX_SOURCE = [
	'precision highp float;',

	'attribute vec3 a_position;',
	'attribute vec2 a_coordinates;',

	'varying vec3 v_position;',
	'varying vec2 v_coordinates;',

	'uniform mat4 u_projectionMatrix;',
	'uniform mat4 u_viewMatrix;',

	'uniform float u_size;',
	'uniform float u_geometrySize;',

	'uniform sampler2D u_displacementMap;',

	'void main (void) {',
		'vec3 position = a_position + texture2D(u_displacementMap, a_coordinates).rgb * (u_geometrySize / u_size);',

		'v_position = position;',
		'v_coordinates = a_coordinates;',

		'gl_Position = u_projectionMatrix * u_viewMatrix * vec4(position, 1.0);',
	'}'
].join('\n');

var OCEAN_FRAGMENT_SOURCE = [
	'precision highp float;',

	'varying vec2 v_coordinates;',
	'varying vec3 v_position;',

	'uniform sampler2D u_displacementMap;',
	'uniform sampler2D u_normalMap;',

	'uniform vec3 u_cameraPosition;',

	'uniform vec3 u_oceanColor;',
	'uniform vec3 u_skyColor;',
	'uniform float u_exposure;',

	'uniform vec3 u_sunDirection;',

	'vec3 hdr (vec3 color, float exposure) {',
		'return 1.0 - exp(-color * exposure);',
	'}',

	'void main (void) {',
		'vec3 normal = texture2D(u_normalMap, v_coordinates).rgb;',

		'vec3 view = normalize(u_cameraPosition - v_position);',
		'float fresnel = 0.02 + 0.98 * pow(1.0 - dot(normal, view), 5.0);',
		'vec3 sky = fresnel * u_skyColor;',

		'float diffuse = clamp(dot(normal, normalize(u_sunDirection)), 0.0, 1.0);',
		'vec3 water = (1.0 - fresnel) * u_oceanColor * u_skyColor * diffuse;',

		'vec3 color = sky + water;',

		'gl_FragColor = vec4(hdr(color, u_exposure), 1.0);',
	'}'
].join('\n');

var Simulator = function (canvas, width, height, viewMatrix, cameraPosition) {
	canvas.width = width;
	canvas.height = height;

	var gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

	var windX = INITIAL_WIND[0],
		windY = INITIAL_WIND[1],
		size = INITIAL_SIZE,
		choppiness = INITIAL_CHOPPINESS;

	gl.getExtension('OES_texture_float');
	gl.getExtension('OES_texture_float_linear');

	gl.clearColor.apply(gl, CLEAR_COLOR);
	gl.enable(gl.DEPTH_TEST);

	var fullscreenVertexShader = buildShader(gl, gl.VERTEX_SHADER, FULLSCREEN_VERTEX_SOURCE);

	var horizontalSubtransformProgram = buildProgramWrapper(gl, fullscreenVertexShader, 
		buildShader(gl, gl.FRAGMENT_SHADER, '#define HORIZONTAL \n' + SUBTRANSFORM_FRAGMENT_SOURCE), {'a_position': 0});
	gl.useProgram(horizontalSubtransformProgram.program);
	gl.uniform1f(horizontalSubtransformProgram.uniformLocations['u_transformSize'], RESOLUTION);

	var verticalSubtransformProgram = buildProgramWrapper(gl, fullscreenVertexShader, 
		buildShader(gl, gl.FRAGMENT_SHADER, SUBTRANSFORM_FRAGMENT_SOURCE), {'a_position': 0});
	gl.useProgram(verticalSubtransformProgram.program);
	gl.uniform1f(verticalSubtransformProgram.uniformLocations['u_transformSize'], RESOLUTION);
	
	var initialSpectrumProgram = buildProgramWrapper(gl, fullscreenVertexShader, 
		buildShader(gl, gl.FRAGMENT_SHADER, INITIAL_SPECTRUM_FRAGMENT_SOURCE), {'a_position': 0});
	gl.useProgram(initialSpectrumProgram.program);
	gl.uniform1f(initialSpectrumProgram.uniformLocations['u_resolution'], RESOLUTION);
	gl.uniform2f(initialSpectrumProgram.uniformLocations['u_wind'], windX, windY);
	gl.uniform1f(initialSpectrumProgram.uniformLocations['u_size'], size);

	var phaseProgram = buildProgramWrapper(gl, fullscreenVertexShader, 
		buildShader(gl, gl.FRAGMENT_SHADER, PHASE_FRAGMENT_SOURCE), {'a_position': 0});
	gl.useProgram(phaseProgram.program);
	gl.uniform1f(phaseProgram.uniformLocations['u_resolution'], RESOLUTION);
	gl.uniform1f(phaseProgram.uniformLocations['u_size'], size);

	var spectrumProgram = buildProgramWrapper(gl, fullscreenVertexShader, 
		buildShader(gl, gl.FRAGMENT_SHADER, SPECTRUM_FRAGMENT_SOURCE), {'a_position': 0});
	gl.useProgram(spectrumProgram.program);
	gl.uniform1i(spectrumProgram.uniformLocations['u_initialSpectrum'], INITIAL_SPECTRUM_UNIT);
	gl.uniform1f(spectrumProgram.uniformLocations['u_resolution'], RESOLUTION);
	gl.uniform1f(spectrumProgram.uniformLocations['u_size'], size);
	gl.uniform1f(spectrumProgram.uniformLocations['u_choppiness'], choppiness);

	var normalMapProgram = buildProgramWrapper(gl, fullscreenVertexShader, 
		buildShader(gl, gl.FRAGMENT_SHADER, NORMAL_MAP_FRAGMENT_SOURCE), {'a_position': 0});
	gl.useProgram(normalMapProgram.program);
	gl.uniform1i(normalMapProgram.uniformLocations['u_displacementMap'], DISPLACEMENT_MAP_UNIT);
	gl.uniform1f(normalMapProgram.uniformLocations['u_resolution'], RESOLUTION);
	gl.uniform1f(normalMapProgram.uniformLocations['u_size'], size);

	var oceanProgram = buildProgramWrapper(gl,
		buildShader(gl, gl.VERTEX_SHADER, OCEAN_VERTEX_SOURCE),
		buildShader(gl, gl.FRAGMENT_SHADER, OCEAN_FRAGMENT_SOURCE), {
			'a_position': 0,
			'a_coordinates': OCEAN_COORDINATES_UNIT
	});
	gl.useProgram(oceanProgram.program);
	gl.uniform1f(oceanProgram.uniformLocations['u_geometrySize'], GEOMETRY_SIZE);
	gl.uniform1i(oceanProgram.uniformLocations['u_displacementMap'], DISPLACEMENT_MAP_UNIT);
	gl.uniform1i(oceanProgram.uniformLocations['u_normalMap'], NORMAL_MAP_UNIT);
	gl.uniform3f(oceanProgram.uniformLocations['u_oceanColor'], OCEAN_COLOR[0], OCEAN_COLOR[1], OCEAN_COLOR[2]);
	gl.uniform3f(oceanProgram.uniformLocations['u_skyColor'], SKY_COLOR[0], SKY_COLOR[1], SKY_COLOR[2]);
	gl.uniform3f(oceanProgram.uniformLocations['u_sunDirection'], SUN_DIRECTION[0], SUN_DIRECTION[1], SUN_DIRECTION[2]);
	gl.uniform1f(oceanProgram.uniformLocations['u_exposure'], EXPOSURE);
	gl.uniform1f(oceanProgram.uniformLocations['u_size'], size);
	gl.uniformMatrix4fv(oceanProgram.uniformLocations['u_viewMatrix'], false, viewMatrix);
	gl.uniform3fv(oceanProgram.uniformLocations['u_cameraPosition'], cameraPosition);

	gl.enableVertexAttribArray(0);

	var fullscreenVertexBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, fullscreenVertexBuffer);
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]), gl.STATIC_DRAW);
	
	var xIndex, zIndex, oceanData = [];
	for (zIndex = 0; zIndex < GEOMETRY_RESOLUTION; zIndex += 1) {
		for (xIndex = 0; xIndex < GEOMETRY_RESOLUTION; xIndex += 1) {
			oceanData.push((xIndex * GEOMETRY_SIZE) / (GEOMETRY_RESOLUTION - 1) + GEOMETRY_ORIGIN[0]);
			oceanData.push((0.0));
			oceanData.push((zIndex * GEOMETRY_SIZE) / (GEOMETRY_RESOLUTION - 1) + GEOMETRY_ORIGIN[1]);
			oceanData.push(xIndex / (GEOMETRY_RESOLUTION - 1));
			oceanData.push(zIndex / (GEOMETRY_RESOLUTION - 1));
		}
	}
	
	var oceanIndices = [];
	for (zIndex = 0; zIndex < GEOMETRY_RESOLUTION - 1; zIndex += 1) {
		for (xIndex = 0; xIndex < GEOMETRY_RESOLUTION - 1; xIndex += 1) {
			var topLeft = zIndex * GEOMETRY_RESOLUTION + xIndex,
				topRight = topLeft + 1,
				bottomLeft = topLeft + GEOMETRY_RESOLUTION,
				bottomRight = bottomLeft + 1;

			oceanIndices.push(topLeft);
			oceanIndices.push(bottomLeft);
			oceanIndices.push(bottomRight);
			oceanIndices.push(bottomRight);
			oceanIndices.push(topRight);
			oceanIndices.push(topLeft);
		}
	}

	var oceanBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, oceanBuffer);
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(oceanData), gl.STATIC_DRAW);
	gl.vertexAttribPointer(OCEAN_COORDINATES_UNIT, 2, gl.FLOAT, false, 5 * SIZE_OF_FLOAT, 3 * SIZE_OF_FLOAT);

	var oceanIndexBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, oceanIndexBuffer);
	gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(oceanIndices), gl.STATIC_DRAW);

	var initialSpectrumTexture = buildTexture(gl, INITIAL_SPECTRUM_UNIT, gl.RGBA, gl.FLOAT, RESOLUTION, RESOLUTION, null, gl.REPEAT, gl.REPEAT, gl.NEAREST, gl.NEAREST),
		pongPhaseTexture = buildTexture(gl, PONG_PHASE_UNIT, gl.RGBA, gl.FLOAT, RESOLUTION, RESOLUTION, null, gl.CLAMP_TO_EDGE, gl.CLAMP_TO_EDGE, gl.NEAREST, gl.NEAREST),
		spectrumTexture = buildTexture(gl, SPECTRUM_UNIT, gl.RGBA, gl.FLOAT, RESOLUTION, RESOLUTION, null, gl.CLAMP_TO_EDGE, gl.CLAMP_TO_EDGE, gl.NEAREST, gl.NEAREST),
		displacementMap = buildTexture(gl, DISPLACEMENT_MAP_UNIT, gl.RGBA, gl.FLOAT, RESOLUTION, RESOLUTION, null, gl.CLAMP_TO_EDGE, gl.CLAMP_TO_EDGE, gl.LINEAR, gl.LINEAR),
		normalMap = buildTexture(gl, NORMAL_MAP_UNIT, gl.RGBA, gl.FLOAT, RESOLUTION, RESOLUTION, null, gl.CLAMP_TO_EDGE, gl.CLAMP_TO_EDGE, gl.LINEAR, gl.LINEAR),
		pingTransformTexture = buildTexture(gl, PING_TRANSFORM_UNIT, gl.RGBA, gl.FLOAT, RESOLUTION, RESOLUTION, null, gl.CLAMP_TO_EDGE, gl.CLAMP_TO_EDGE, gl.NEAREST, gl.NEAREST),
		pongTransformTexture = buildTexture(gl, PONG_TRANSFORM_UNIT, gl.RGBA, gl.FLOAT, RESOLUTION, RESOLUTION, null, gl.CLAMP_TO_EDGE, gl.CLAMP_TO_EDGE, gl.NEAREST, gl.NEAREST);

	var pingPhase = true;

	var phaseArray = new Float32Array(RESOLUTION * RESOLUTION * 4);
	for (var i = 0; i < RESOLUTION; i += 1) {
		for (var j = 0; j < RESOLUTION; j += 1) {
			phaseArray[i * RESOLUTION * 4 + j * 4] = Math.random() * 2.0 * Math.PI;
			phaseArray[i * RESOLUTION * 4 + j * 4 + 1] = 0;
			phaseArray[i * RESOLUTION * 4 + j * 4 + 2] = 0;
			phaseArray[i * RESOLUTION * 4 + j * 4 + 3] = 0;
		}
	}
	var pingPhaseTexture = buildTexture(gl, PING_PHASE_UNIT, gl.RGBA, gl.FLOAT, RESOLUTION, RESOLUTION, phaseArray, gl.CLAMP_TO_EDGE, gl.CLAMP_TO_EDGE, gl.NEAREST, gl.NEAREST);

	//changing framebuffers faster than changing attachments in WebGL
	var initialSpectrumFramebuffer = buildFramebuffer(gl, initialSpectrumTexture),
		pingPhaseFramebuffer = buildFramebuffer(gl, pingPhaseTexture),
		pongPhaseFramebuffer = buildFramebuffer(gl, pongPhaseTexture),
		spectrumFramebuffer = buildFramebuffer(gl, spectrumTexture),
		displacementMapFramebuffer = buildFramebuffer(gl, displacementMap),
		normalMapFramebuffer = buildFramebuffer(gl, normalMap),
		pingTransformFramebuffer = buildFramebuffer(gl, pingTransformTexture),
		pongTransformFramebuffer = buildFramebuffer(gl, pongTransformTexture);

	gl.bindBuffer(gl.ARRAY_BUFFER, fullscreenVertexBuffer);
	gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

	gl.bindFramebuffer(gl.FRAMEBUFFER, initialSpectrumFramebuffer);
	gl.useProgram(initialSpectrumProgram.program);
	gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

	this.resize = function (width, height) {
		canvas.width = width;
		canvas.height = height;
	};

	this.render = function (deltaTime, projectionMatrix) {
		gl.viewport(0, 0, RESOLUTION, RESOLUTION);
		gl.disable(gl.DEPTH_TEST);

		gl.bindBuffer(gl.ARRAY_BUFFER, fullscreenVertexBuffer);
		gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

		//store phases separately to ensure continuity of waves during parameter editing
		gl.useProgram(phaseProgram.program);
		gl.bindFramebuffer(gl.FRAMEBUFFER, pingPhase ? pongPhaseFramebuffer : pingPhaseFramebuffer);
		gl.uniform1i(phaseProgram.uniformLocations['u_phases'], pingPhase ? PING_PHASE_UNIT : PONG_PHASE_UNIT);
		pingPhase = !pingPhase;
		gl.uniform1f(phaseProgram.uniformLocations['u_deltaTime'], deltaTime);
		gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

		gl.useProgram(spectrumProgram.program);
		gl.bindFramebuffer(gl.FRAMEBUFFER, spectrumFramebuffer);
		gl.uniform1i(spectrumProgram.uniformLocations['u_phases'], pingPhase ? PING_PHASE_UNIT : PONG_PHASE_UNIT);
		gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

		var subtransformProgram = horizontalSubtransformProgram;
		gl.useProgram(horizontalSubtransformProgram.program);

		//GPU FFT using Stockham formulation
		var iterations = log2(RESOLUTION) * 2;
		for (var i = 0; i < iterations; i += 1) {
			if (i === 0) {
				gl.bindFramebuffer(gl.FRAMEBUFFER, pingTransformFramebuffer);
				gl.uniform1i(subtransformProgram.uniformLocations['u_input'], SPECTRUM_UNIT);
			} else if (i === iterations - 1) {
				gl.bindFramebuffer(gl.FRAMEBUFFER, displacementMapFramebuffer);
				gl.uniform1i(subtransformProgram.uniformLocations['u_input'], (iterations % 2 === 0) ? PING_TRANSFORM_UNIT : PONG_TRANSFORM_UNIT);
			} else if (i % 2 === 1) {
				gl.bindFramebuffer(gl.FRAMEBUFFER, pongTransformFramebuffer);
				gl.uniform1i(subtransformProgram.uniformLocations['u_input'], PING_TRANSFORM_UNIT);
			} else {
				gl.bindFramebuffer(gl.FRAMEBUFFER, pingTransformFramebuffer);
				gl.uniform1i(subtransformProgram.uniformLocations['u_input'], PONG_TRANSFORM_UNIT);
			}

			if (i === iterations / 2) {
				subtransformProgram = verticalSubtransformProgram;
				gl.useProgram(verticalSubtransformProgram.program);
			}

			gl.uniform1f(subtransformProgram.uniformLocations['u_subtransformSize'], Math.pow(2,(i % (iterations / 2)) + 1));
			gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
		}

		gl.bindFramebuffer(gl.FRAMEBUFFER, normalMapFramebuffer);
		gl.useProgram(normalMapProgram.program);
		gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.viewport(0, 0, canvas.width, canvas.height);
		gl.enable(gl.DEPTH_TEST);
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

		gl.enableVertexAttribArray(OCEAN_COORDINATES_UNIT);

		gl.bindBuffer(gl.ARRAY_BUFFER, oceanBuffer);
		gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 5 * SIZE_OF_FLOAT, 0);

		gl.useProgram(oceanProgram.program);
		gl.uniformMatrix4fv(oceanProgram.uniformLocations['u_projectionMatrix'], false, projectionMatrix);
		gl.drawElements(gl.TRIANGLES, oceanIndices.length, gl.UNSIGNED_SHORT, 0);

		gl.disableVertexAttribArray(OCEAN_COORDINATES_UNIT);
		
	};

};

var main = function () {
	var simulatorCanvas = document.getElementById(SIMULATOR_CANVAS_ID);

	var camera = new Camera(),
		projectionMatrix = makePerspectiveMatrix(new Float32Array(16), FOV, MIN_ASPECT, NEAR, FAR);
	
	var simulator = new Simulator(simulatorCanvas, window.innerWidth, window.innerHeight, camera.getViewMatrix(), camera.getPosition());

	var onresize = function () {
		var windowWidth = window.innerWidth,
		windowHeight = window.innerHeight;

		makePerspectiveMatrix(projectionMatrix, FOV, windowWidth / windowHeight, NEAR, FAR);
		simulator.resize(windowWidth, windowHeight);
	};

	window.addEventListener('resize', onresize);
	onresize();

	var lastTime = 0;
	var timeStep = 1000 / 30;

	var render = function render (currentTime) {
		var deltaTime = (currentTime - lastTime) || 0.0;

		if (deltaTime >= timeStep) {
			lastTime = currentTime;

			simulator.render(deltaTime, projectionMatrix);
		}

		window.requestAnimationFrame(render);
	};
	render(0.0);
};

if (hasWebGLSupportWithExtensions(['OES_texture_float', 'OES_texture_float_linear'])) {
	main();
}

}());
