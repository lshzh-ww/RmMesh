def jet(x):
    r=int((1.5-4.*abs(x-0.75))*255.)
    g=int((1.5-4.*abs(x-0.5))*255.)
    b=int((1.5-4.*abs(x-0.25))*255.)
    if r<0:
        r=0
    elif r>255:
        r=255
    
    if g<0:
        g=0
    elif g>255:
        g=255
    
    if b<0:
        b=0
    elif b>255:
        b=255
    return [r,g,b]
