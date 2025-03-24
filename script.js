document.getElementById('getOtpButton').addEventListener('click', function() {
    const phoneNumber = document.getElementById('phoneNumber').value;
    if (phoneNumber) {
        alert(`OTP sent to ${phoneNumber}`);
    } else {
        alert('Please enter a valid phone number.');
    }
});

document.getElementById('chooseLanguageButton').addEventListener('click', function() {
    const languages = ['English', 'Spanish', 'French', 'German'];
    const chosenLanguage = prompt(`Choose a language:\n${languages.join(', ')}`);
    if (languages.includes(chosenLanguage)) {
        alert(`You have chosen: ${chosenLanguage}`);
    } else {
        alert('Invalid language selected.');
    }
});

document.getElementById('loginButton').addEventListener('click', function() {
    alert('Login button clicked!');
});