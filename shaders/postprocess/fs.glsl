#version 400

in vec2 uv;

uniform sampler2D TexColor;
uniform sampler2D TexDepth;
uniform float screen_width;
uniform float screen_height;
uniform vec2 near_far;
uniform float elapsed;
uniform float zoom;

out vec4 color_out;

uniform float debug;

float LinearizeDepth(float z)
{
	float n = near_far.x; // camera z near
  	float f = near_far.y; // camera z far
  	return (2.0 * n) / (f + n - z * (f - n));
}

float rand(vec2 c){
	return fract(sin(dot(c.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

#define PI 3.14159265358979323846
float noise(vec2 p, float freq ){
	float unit = 1.0/freq;
	vec2 ij = floor(p/unit);
	vec2 xy = mod(p,unit)/unit;
	//xy = 3.*xy*xy-2.*xy*xy*xy;
	xy = .5*(1.-cos(PI*xy));
	float a = rand((ij+vec2(0.,0.)));
	float b = rand((ij+vec2(1.,0.)));
	float c = rand((ij+vec2(0.,1.)));
	float d = rand((ij+vec2(1.,1.)));
	float x1 = mix(a, b, xy.x);
	float x2 = mix(c, d, xy.x);
	return mix(x1, x2, xy.y);
}

vec2 fade(vec2 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}

float cnoise(vec2 P){
  vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  Pi = mod(Pi, 289.0); // To avoid truncation effects in permutation
  vec4 ix = Pi.xzxz;
  vec4 iy = Pi.yyww;
  vec4 fx = Pf.xzxz;
  vec4 fy = Pf.yyww;
  vec4 i = permute(permute(ix) + iy);
  vec4 gx = 2.0 * fract(i * 0.0243902439) - 1.0; // 1/41 = 0.024...
  vec4 gy = abs(gx) - 0.5;
  vec4 tx = floor(gx + 0.5);
  gx = gx - tx;
  vec2 g00 = vec2(gx.x,gy.x);
  vec2 g10 = vec2(gx.y,gy.y);
  vec2 g01 = vec2(gx.z,gy.z);
  vec2 g11 = vec2(gx.w,gy.w);
  vec4 norm = 1.79284291400159 - 0.85373472095314 * 
    vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
  g00 *= norm.x;
  g01 *= norm.y;
  g10 *= norm.z;
  g11 *= norm.w;
  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));
  vec2 fade_xy = fade(Pf.xy);
  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return 2.3 * n_xy;
}

/*float baseYarn(float t, float s){
	float base = max(0,cos(t*200)+0.6)/1.6;
	base = clamp(0,1,base);
	base -= max(0,cos(s*20));
	base = clamp(0,1,base);
	return base;
}*/


float baseYarn(float t, float s, float freq){

	t*=3.0;
	
	float sDown = s;
	if(cos(t*freq/2) > 0)
		sDown+=PI/freq;

	float base = max(0,cos(t*freq)+1.2)/2.2;
	base = clamp(base,0,1);
	float down = max(0,cos(sDown*freq)+1.0)/2.0;
	down = clamp(down,0,1);
	float horzYarn = max(0,cos(s*2*freq)+0.6)/1.6;
	horzYarn = clamp(horzYarn,0,1);

	down = pow(down,2);
	down = (down +0.1) / 1.1;
	
	/*if(debug == 0 || debug >= 8)
		base = max(base * down, horzYarn*0.7);

	if(debug == 1)
		base = base;
	if(debug == 2)
		base = down;
	if(debug == 3)
		base *= down;
	if(debug == 4)
		base = horzYarn;
	if(debug == 5)
		base = max(base * down, horzYarn*0.8);*/

	base = max(base * down, horzYarn*0.7);
			

	base = clamp(base,0.02,1.0); 
	return base;
}

float yarn(vec2 uv, float freq){
	float bruit = cnoise(uv*freq/4);
	float tBruit = uv.x + cos(bruit*2*PI)/(freq*2);
	float sBruit = uv.y + cos(bruit*2*PI)/(freq*8);

	//float yarnHeight =  baseYarn(uv.x,uv.y,freq);
	float yarnHeight = baseYarn(tBruit,uv.y+sBruit,freq);



	return yarnHeight;
}

float filHoriz(float t, float freq){
	float base = max(0,cos(t*3*freq)+0.6)/1.6;
	//float base = max(0,sin(t*freq*PI*2-PI/2)+1)/2.0f;
	//float base = max(0,cos(t*freq*3)+1)/2.0f;

	return clamp(base,0,1);
	//return sin(t*freq*PI*2-PI/2);
}

float fil(vec2 uv, vec2 dir, float freq){
	vec2 dirPerp = vec2(dir.y,-dir.x);
	float t = dot(dirPerp,uv);
	return filHoriz(t,freq);
}

/*vec3 dfil(vec2 uv, vec2 dir, float freq){
	dir = normalize(dir);
	vec2 dirPerp = vec2(dir.y,-dir.x);
	float t = dot(dirPerp,uv);
	float dt =	(0.001/freq)/screen_width;
	float dz_dt = filHoriz(t+dt,freq) - filHoriz(t,freq);


	vec3 A =  normalize(vec3(dirPerp*dt,dz_dt/10));
	vec3 B =  vec3(dir,0);
	return cross(A,B); 
}*/

//float coefs[] = {0.1,0.2,0.15,0.23,0.03,0.2,0.14,0.3,0.14,0,11};
/*float patchit(vec2 uv, float freq){
	float yarnHeight = 0;
	float n = 5;
	float bruit;
	for(int i=0;i<n;i++){
		float coeff = cnoise(vec2((4*i+1)*0.23,(3*i+1)*0.13))/2;
		bruit = cnoise(vec2(coeff*0.17*10,uv.y*3));
		bruit = (bruit + 1)/2;
		bruit *= 0.15;	
		if(sign(coeff) < 0)
			bruit *= 0.015;	

		float newYDir = bruit;
		vec2 dir = normalize(vec2(coeff,newYDir*sign(coeff)));
		//vec2 uv2 = vec2(uv.x,uv.y);
		//yarnHeight = max(yarnHeight,fil(uv2,dir,freq));
		//yarnHeight = yarnHeight*(fil(uv2,dir,freq)+0.5f)/1.5;

		yarnHeight = max(pow(fil(uv,dir,freq),4),yarnHeight);
	}
		
	return clamp(yarnHeight,0,1);
}*/

float patchit(vec2 uv, float freq){
	float yarnHeight = 0;
	
	float n = 1;
	for(int i=0;i<n;i++){
		float coeff = rand(vec2((i+1)*0.1,0.17));
		float newYDir = cnoise(vec2(0.17,uv.y+coeff));
		float decUvY = rand(vec2(i*2.14,i*0.21));
		//decUvY += sin(elapsed)/100;
		//newYDir = (newYDir + 1)/2;
		newYDir *= 0.5;	

		vec2 dir = normalize(vec2(newYDir,1));
		vec2 uv2 = vec2(uv.x,uv.y+decUvY);
		//yarnHeight = max(yarnHeight,fil(uv2,dir,freq));
		//yarnHeight = yarnHeight*(fil(uv2,dir,freq)+0.5f)/1.5;

		yarnHeight = max(pow(fil(uv2,dir,freq),3),yarnHeight)/2;
	}

	n = 5;
	for(int i=0;i<n;i++){
		float coeff = rand(vec2((i+1)*0.1,0.17));
		float newYDir = cnoise(vec2(0.17,uv.y+coeff));
		float decUvY = rand(vec2(i*2.14,i*0.21));
		//decUvY += sin(elapsed)/100;
		//newYDir = (newYDir + 1)/2;
		newYDir *= 0.5;	

		vec2 dir = normalize(vec2(1,newYDir));
		vec2 uv2 = vec2(uv.x,uv.y+decUvY);
		//yarnHeight = max(yarnHeight,fil(uv2,dir,freq));
		//yarnHeight = yarnHeight*(fil(uv2,dir,freq)+0.5f)/1.5;

		yarnHeight = max(pow(fil(uv2,dir,freq),5),yarnHeight);
		/*yarnHeight = mix(yarnHeight*(1-pow(yarnHeight,0.1)),
			pow(fil(uv2,dir,freq),5),
			pow(fil(uv2,dir,freq),5));*/
	}
		
	return clamp(yarnHeight,0,1);
}

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

void main (void)
{
	float screen_width_zoom = screen_width/zoom;
	float screen_height_zoom = screen_height/zoom;

	float xstep = 1.0/screen_width;
	float ystep = 1.0/screen_height;
	float ratio = screen_width / screen_height;

	vec3 colorDessin = texture2D( TexColor , uv ).rgb;
	float saturation = rgb2hsv(colorDessin.xyz).y; 

	//Ombrage cerné
	float sizeCerne = 5;
	vec2 decal = vec2(xstep,0) * sizeCerne;
	vec3 colLookup = texture2D( TexColor , uv + decal).rgb;
	float difColTour = length(colLookup - colorDessin);//rgb2hsv(colLookup.xyz).y; 

	decal = vec2(-xstep,0) * sizeCerne;
	colLookup = texture2D( TexColor , uv + decal).rgb;
	difColTour += length(colLookup - colorDessin);//rgb2hsv(colLookup.xyz).y; 

	decal = vec2(0,ystep) * sizeCerne;
	colLookup = texture2D( TexColor , uv + decal).rgb;
	difColTour += length(colLookup - colorDessin);//rgb2hsv(colLookup.xyz).y; 

	decal = vec2(0,-ystep) * sizeCerne;
	colLookup = texture2D( TexColor , uv + decal).rgb;
	difColTour += length(colLookup - colorDessin);//rgb2hsv(colLookup.xyz).y; 

	decal = vec2(xstep,ystep) * sizeCerne;
	colLookup = texture2D( TexColor , uv + decal).rgb;
	difColTour += length(colLookup - colorDessin) * 0.7;//rgb2hsv(colLookup.xyz).y; 

	decal = vec2(-xstep,-ystep) * sizeCerne;
	colLookup = texture2D( TexColor , uv + decal).rgb;
	difColTour += length(colLookup - colorDessin) * 0.7;//rgb2hsv(colLookup.xyz).y; 

	decal = vec2(-xstep,ystep) * sizeCerne;
	colLookup = texture2D( TexColor , uv + decal).rgb;
	difColTour += length(colLookup - colorDessin) * 0.7;//rgb2hsv(colLookup.xyz).y; 

	decal = vec2(xstep,-ystep) * sizeCerne;
	colLookup = texture2D( TexColor , uv + decal).rgb;
	difColTour += length(colLookup - colorDessin) * 0.7;//rgb2hsv(colLookup.xyz).y; 

	sizeCerne = 10;
	decal = vec2(xstep,0) * sizeCerne;
	colLookup = texture2D( TexColor , uv + decal).rgb;
	difColTour += length(colLookup - colorDessin);//rgb2hsv(colLookup.xyz).y; 

	decal = vec2(-xstep,0) * sizeCerne;
	colLookup = texture2D( TexColor , uv + decal).rgb;
	difColTour += length(colLookup - colorDessin);//rgb2hsv(colLookup.xyz).y; 

	decal = vec2(0,ystep) * sizeCerne;
	colLookup = texture2D( TexColor , uv + decal).rgb;
	difColTour += length(colLookup - colorDessin);//rgb2hsv(colLookup.xyz).y; 

	decal = vec2(0,-ystep) * sizeCerne;
	colLookup = texture2D( TexColor , uv + decal).rgb;
	difColTour += length(colLookup - colorDessin);//rgb2hsv(colLookup.xyz).y; 

	decal = vec2(xstep,ystep) * sizeCerne;
	colLookup = texture2D( TexColor , uv + decal).rgb;
	difColTour += length(colLookup - colorDessin) * 0.7;//rgb2hsv(colLookup.xyz).y; 

	decal = vec2(-xstep,-ystep) * sizeCerne;
	colLookup = texture2D( TexColor , uv + decal).rgb;
	difColTour += length(colLookup - colorDessin) * 0.7;//rgb2hsv(colLookup.xyz).y; 

	decal = vec2(-xstep,ystep) * sizeCerne;
	colLookup = texture2D( TexColor , uv + decal).rgb;
	difColTour += length(colLookup - colorDessin) * 0.7;//rgb2hsv(colLookup.xyz).y; 

	decal = vec2(xstep,-ystep) * sizeCerne;
	colLookup = texture2D( TexColor , uv + decal).rgb;
	difColTour += length(colLookup - colorDessin) * 0.7;//rgb2hsv(colLookup.xyz).y; 

	float difCol = difColTour/13.6; 

	float freq = screen_width/3;
		
	//Trame de fond
	vec3 colorFond = vec3(255, 255, 255)/255.0;
	float yarnHeight = yarn(vec2(uv.x,uv.y/ratio),freq);
	colorFond = yarnHeight*colorFond;

	float patchHeight =  patchit(vec2(uv.x,uv.y/ratio),freq);

	vec3 colorPatch = mix(colorFond * (1-pow(patchHeight,0.5)), colorDessin, patchHeight);
	
	float mixFondPatch = (pow(saturation,0.6) + 0.7)/1.7; 
	mixFondPatch = clamp(mixFondPatch + sqrt(3)-length(colorDessin),0,1); 
	vec3 color = mix(colorFond,colorPatch,mixFondPatch);

	float ombre = max((1-saturation),(1-difCol));// * (1-saturation+1);

	color *= (ombre+0.3)/1.3; 
	

	//s
	//color.rgb = mix(color.rgb,patchHeight*colorFond.rgb,1-(saturation));
	saturation = saturation;
	//color = vec3(difCol,difCol,difCol);
	color_out = vec4(color,1.0);
}