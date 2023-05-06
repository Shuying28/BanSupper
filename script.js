//animation on scroll
$(window).scroll(function () {
  $(".animation").each(function () {
    var position = $(this).offset().top;
    var windowHeight = $(window).height();
    if (position < $(window).scrollTop() + windowHeight) {
      $(this).addClass("show");
    }
  });
});

//animation on load
$(document).ready(function () {
  $('.animation-on-load').each(function (i) {
    setTimeout(function () {
      $('.animation-on-load').eq(i).css({
        'opacity': '1',
        'transform': 'translateY(0)'
      });
    }, 400 * (i + 1));
  });
});

//navbar
$('.expandHome').mouseover(function() {
  $('.sub-home').css({
        'display': 'block'
    }); 
});
$('.subnavbtn').mouseover(function() {
  $('.sub-home').css({
        'display': 'none'
    }); 
});

$('#trapezoid').mouseleave(function() {
  $('#trapezoid').css({
        'margin-top': '-53px'
    }); 
    $('.sub-home').css({
        'display': 'none'
    }); 
}).mouseenter(function() {
  $('#trapezoid').css({
        'margin-top': '0px'
    }); 
});
       
