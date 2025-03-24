// SVG Animation
gsap.fromTo("#svg", { strokeDashoffset: 4500 }, {
    strokeDashoffset: 0,
    duration: 5,
    ease: "power4.inOut"
});
// Fade-in Text Animation
gsap.fromTo("#logoText", { opacity: 0 }, {
    opacity: 1,
    duration: 5,
    delay: 2,
    ease: "power4.out"
});
// Hide Loading Page Animation 
gsap.to(".lpage", {
    opacity: 0,
    duration: 3,
    delay: 3,
    ease: "power4.out"
 });
gsap.to("#svg", {
    strokeDashoffset: 0,
    duration: 3.5,  
    onComplete: function() {
        setTimeout(() => {
            window.location.href = "sym.html"; 
        }, 1000); // Adjust delay if needed
    }
});
