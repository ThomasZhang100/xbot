
import pyglet

x=[100,100,600,600,1100,1100]
y=[100,400,200,600,100,800]
animation = pyglet.image.load_animation('tenor.gif')

animSprite=[]
for i in range(0,len(x)):
    animSprite.append(pyglet.sprite.Sprite(animation))
    animSprite[i].position=(x[i],y[i])
 
window = pyglet.window.Window(width=1600, height=1200)
 
r,g,b,alpha = 1,1,1,1
 
 
pyglet.gl.glClearColor(r,g,b,alpha)

@window.event
def on_draw():
    window.clear()
    for i in range(0,len(x)):
        animSprite[i].position=(x[i],y[i])
        animSprite[i].draw()
        x[i]=x[i]+40
        if x[i]>1200:
            x[i]=0
 
pyglet.app.run()

