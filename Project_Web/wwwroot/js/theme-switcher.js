const themeToggle = document.getElementById('themeSwitcher');
const savedTheme = getCookie('theme') || 'light';

setTheme(savedTheme);
updateLogo(savedTheme);
themeToggle.checked = savedTheme === 'dark';

themeToggle.addEventListener('change', () => {
    const newTheme = themeToggle.checked ? 'dark' : 'light';
    setTheme(newTheme);
    updateLogo(newTheme);
    // saveUserTheme(newTheme);
    document.cookie = `theme=${newTheme}; path=/; max-age=31536000`;
});

function updateLogo(theme) {
    var small = document.getElementById('small-logo');
    var big = document.getElementById('big-logo');
    var small_path = "/logos/small-logo-" + theme + ".png";
    var big_path = "/logos/big-logo-" + theme + ".png";
    console.log(small_path);
    console.log(big_path);
    small.src = small_path;
    big.src = big_path;
}
function setTheme(theme) {
    document.documentElement.setAttribute('data-bs-theme', theme);
}

function getCookie(name) {
    const match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
    return match ? match[2] : null;
}

