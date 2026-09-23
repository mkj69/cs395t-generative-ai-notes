const menuButton = document.querySelector('.menu-button');
const siteNav = document.querySelector('.site-nav');

if (menuButton && siteNav) {
  menuButton.addEventListener('click', () => {
    const isOpen = menuButton.getAttribute('aria-expanded') === 'true';
    menuButton.setAttribute('aria-expanded', String(!isOpen));
    siteNav.classList.toggle('is-open', !isOpen);
  });
}

const noteList = document.querySelector('#note-list');
const searchInput = document.querySelector('#note-search');
const typeFilter = document.querySelector('#type-filter');
const noteCount = document.querySelector('#note-count');

if (noteList && searchInput && typeFilter && noteCount) {
  let notes = [];

  const renderNotes = () => {
    const query = searchInput.value.trim().toLowerCase();
    const type = typeFilter.value;
    const visible = notes.filter((note) => {
      const haystack = [note.title, note.question, note.typeLabel, ...note.tags].join(' ').toLowerCase();
      return (type === 'all' || note.type === type) && haystack.includes(query);
    });

    noteCount.textContent = `${visible.length} structural item${visible.length === 1 ? '' : 's'}`;

    if (!visible.length) {
      noteList.innerHTML = '<p class="empty-state">Nothing matches this filter. No course notes have been published yet.</p>';
      return;
    }

    noteList.innerHTML = visible.map((note, index) => `
      <article class="note-card">
        <div class="note-card__number">${String(index + 1).padStart(2, '0')}</div>
        <div class="note-card__body">
          <div class="note-card__meta">
            <span>${note.typeLabel}</span>
            <span>${note.status}</span>
            <span>${note.date}</span>
          </div>
          <h3><a href="${note.url}">${note.title}</a></h3>
          <p>${note.question}</p>
          <ul class="tag-list">${note.tags.map((tag) => `<li>${tag}</li>`).join('')}</ul>
        </div>
        <div class="note-card__action">
          <span>${note.disclaimer}</span>
          <a href="${note.url}" aria-label="Open ${note.title}">Open →</a>
        </div>
      </article>
    `).join('');
  };

  fetch('data/notes.json')
    .then((response) => {
      if (!response.ok) throw new Error('Could not load note index.');
      return response.json();
    })
    .then((data) => {
      notes = data;
      renderNotes();
    })
    .catch(() => {
      noteCount.textContent = 'Index unavailable';
      noteList.innerHTML = '<p class="empty-state">The note index could not be loaded. Try serving the site through a local web server.</p>';
    });

  searchInput.addEventListener('input', renderNotes);
  typeFilter.addEventListener('change', renderNotes);
}
