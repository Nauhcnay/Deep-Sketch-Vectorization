'''
## Install
python3 -m venv .venv
source .venv/bin/activate
pip install svgpathtools numpy
## Run
python3 svg2scap.py input.svg
'''

from svgpathtools import Document, svg2paths, Line
import numpy as np

def svg2scap( path_to_svg, output_path, spacing = 2 ):
    '''
    Given:
        path_to_svg: A path to an SVG file.
        output_path: A path to save a .scap file.
        spacing (optional): The arc length spacing along smooth curves.
    
    Saves the SVG path data as a .scap file.
    
    Ignores <svg> width or height attributes.
    '''
    flatpaths = None
    paths = None
    try:
        doc = Document( path_to_svg )
        flatpaths = doc.flatten_all_paths()
        print( "Using svgpathtools.Document" )
    except:
        paths, _ = svg2paths( path_to_svg,
            convert_circles_to_paths = False,
            convert_ellipses_to_paths = False,
            convert_rectangles_to_paths = False
            )
        flatpaths = paths
        print( "Using svgpathtools.svg2paths" )
    
    linestrips = []
    
    for path in flatpaths:
        last_pt = None
        
        # if this is get by flatten_all_paths, then we just need to get the first item
        # if paths == None: path =  path[0]
        
        for seg in path:
            ## Make sure this segment is connected to the previous one.
            ## If not, start a new one.
            ## I checked and it doesn't look like svgpathtools tells us when a Move
            ## command happens, so we have to figure it out.
            if last_pt is None or not np.allclose( last_pt, seg.start ):
                linestrips.append( [ pt2tuple( seg.start ) ] )
            
            ## Add points along the current segment.
            if type(seg) is not Line:
                N = int( (seg.length()+spacing-1)/spacing )
                print( f"Tessellating a smooth curve {N} times." )
                for t in np.linspace( 0, 1, N )[1:-1]:
                    linestrips[-1].append( pt2tuple( seg.point(t) ) )
            
            ## Add the last point
            last_pt = seg.end
            linestrips[-1].append( pt2tuple( seg.end ) )
    
    ## Divide by long edge.
    if 'viewBox' in doc.root.attrib:
        import re
        _, _, width, height = [ float(v) for v in re.split( '[ ,]+', doc.root.attrib['viewBox'].strip() ) ]
    elif "width" in doc.root.attrib and "height" in doc.root.attrib:
        width = doc.root.attrib["width"].strip().strip("px")
        height = doc.root.attrib["height"].strip().strip("px")
    else:
        print( "WARNING: No viewBox found in <svg>. Using maximum value of paths." )
        width, height = np.max( [ np.max( linestrip, axis = 0 ) for linestrip in linestrips ], axis = 0 )
    
    ## Convert width and height to integers?
    width, height = np.ceil( ( width, height ) ).astype(int)
    
    with open( output_path, 'w' ) as out:
        out.write( f"#{width}	{height}\n" )
        out.write( "@2\n" )
        for i, linestrip in enumerate( linestrips ):
            out.write( "{\n" )
            out.write( f"	#{i}	0\n" )
            for x,y in linestrip: out.write( f"	{x}	{y}	0\n" )
            out.write( "}\n" )
    
    print( "Saved: ", output_path )

def pt2tuple( p ):
    return p.real, p.imag

if __name__ == '__main__':
    import argparse, pathlib, sys
    
    parser = argparse.ArgumentParser( description = "Convert an SVG file's paths to a scap file." )
    parser.add_argument( "path_to_svg", help = "The SVG file to convert." )
    args = parser.parse_args()
    
    if not pathlib.Path(args.path_to_svg).is_file():
        print( "ERROR: No file at input path:", args.path_to_svg )
        sys.exit(-1)
    
    output_path = pathlib.Path( args.path_to_svg ).with_suffix( '.scap' )
    if output_path.exists():
        print( "ERROR: Output path exists, won't clobber:", output_path )
        sys.exit(-1)
    
    svg2scap( args.path_to_svg, output_path )
