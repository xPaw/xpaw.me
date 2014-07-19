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
			
			element.textContent = element.textContent.replace( ' and', ',' ).replace( '.', ', and stalk corporations.' );
			
			return false;
		}
	};
	
	document.addEventListener( 'keydown', callback, false );
}());
