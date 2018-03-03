#version 400

in vec2 uv;

uniform sampler2D TexColor;
uniform sampler2D TexDepth;
uniform float screen_width;
uniform float screen_height;
uniform vec2 near_far;
uniform float elapsed;

out vec4 color_out;

uniform float debug;

float LinearizeDepth(float z)
{
	float n = near_far.x; // camera z near
  	float f = near_far.y; // camera z far
  	return (2.0 * n) / (f + n - z * (f - n));
}


void main (void)
{
	float xstep = 1.0/screen_width;
	float ystep = 1.0/screen_height;
	float ratio = screen_width / screen_height;

	vec4 color = texture2D( TexColor , uv );

    float dist = 1-(length(uv - vec2(0.5,0.5))*1.42);
    float vignettage = dist;
    vignettage = pow(vignettage,0.3); 

	
	//color = 1-color;
	/*int radius = 2;
    float n = float (( radius + 1) * ( radius + 1));
    vec3 m [4];
    vec3 s [4];
    for (int k = 0; k < 4; ++ k) {
        m[k] = vec3 (0.0);
        s[k] = vec3 (0.0);
    }
    struct Window { int x1 , y1 , x2 , y2; };
    Window W[4] = Window [4](
        Window ( -radius , -radius , 0, 0 ),
        Window ( 0, -radius , radius , 0 ),
        Window ( 0, 0, radius , radius ),
        Window ( -radius , 0, 0, radius )
    );
    for (int k = 0; k < 4; ++ k) {
       for (int j = W[k]. y1; j <= W[k].y2; ++ j) {
            for (int i = W[k].x1; i <= W[k]. x2; ++ i) {
                vec3 c = texture2D (TexColor , uv + vec2(i ,j) / vec2(screen_width,screen_height) ). rgb ;
                m[k] += c;
                s[k] += c * c;
            }
        }
    }
    float min_sigma2 = 1e+2;
    for (int k = 0; k < 4; ++ k) {
        m[k] /= n;
        s[k] = abs (s[k] / n - m[k] * m[k]);
        float sigma2 = s[k].r + s[k].g + s[k].b;
        if ( sigma2 < min_sigma2 ) {
            min_sigma2 = sigma2 ;
            color = vec4 (m[k], 1.0);
        }
    }*/

    color.rgb*=vignettage;
    //Gamma correction
    color.rgb = sqrt(clamp(color.rgb,0,1));

	color_out = vec4(color.rgb,1.0);
}