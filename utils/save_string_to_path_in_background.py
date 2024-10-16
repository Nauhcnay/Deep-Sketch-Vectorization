from threading import Thread

def write_string_to_path( string, path ):
    with open( path, 'w', encoding = 'utf-8' ) as f:
        f.write( string )
    print( "Saved:", path )

def save_string_to_path_in_background( string, path ):
    Thread( target = write_string_to_path, args = ( string, path ) ).start()

if __name__ == '__main__':
    save_string_to_path_in_background( """
This is a test of save_string_to_path_in_background().
Does it work?
""", "foo.txt" )
