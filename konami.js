(function()
{
	var input = '', pattern = '38384040373937396665', callback = function( e )
	{
		input += e.keyCode;
		
		if( input.length > pattern.length )
		{
			input = input.substr( ( input.length - pattern.length ) );
		}
		
		if( input === pattern )
		{
			document.removeEventListener( 'keydown', callback, false );
			
			e.preventDefault();
			
			var element = document.querySelector( 'h2' );
			
			element.textContent = element.textContent.replace( ' and', ',' ).replace( '.', ', and \x73\x74\x61\x6C\x6B \x63\x6F\x72\x70\x6F\x72\x61\x74\x69\x6F\x6E\x73.' );
			
			return false;
		}
	};
	
	document.addEventListener( 'keydown', callback, false );
}());
