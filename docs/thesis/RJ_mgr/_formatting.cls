%% ============================================================
%% Document formatting settings
%% Compliance with Rector’s Ordinance PG ZR 45-2024
%% Required packages
%% ============================================================

%% -------------------- Base class _formatting -------------------
\ProvidesClass{_formatting}
%% A4, 10pt font size, two-sided printing
\LoadClass[10pt,a4paper,twoside]{report}

%% -------------------- Language, fonts and typesetting ----------------------
\RequirePackage{fontspec}
\defaultfontfeatures{Ligatures=TeX}
\RequirePackage{polyglossia}
\setdefaultlanguage{english}
\setotherlanguage{polish}
\RequirePackage[newfloat]{minted}
\RequirePackage{csquotes}
\defaultfontfeatures{Ligatures=TeX, Scale=MatchLowercase}

%% using the Arial typeface from the local fonts/ folder
\setmainfont{Arial}[
  Path=fonts/,
  UprightFont    = arial.ttf,
  BoldFont       = arialbd.ttf,
  ItalicFont     = ariali.ttf,
  BoldItalicFont = arialbi.ttf,
  Scale=1.0
]

%% setting the same family (Arial) as the sans-serif font
\setsansfont{Arial}[
  Path=fonts/,
  UprightFont    = arial.ttf,
  BoldFont       = arialbd.ttf,
  ItalicFont     = ariali.ttf,
  BoldItalicFont = arialbi.ttf,
  Scale=1.0
]

%% mathematics: a dedicated math font (Arial does not include math glyphs)
\RequirePackage{unicode-math}
\setmathfont{Latin Modern Math} % alternatively STIX Two Math / XITS / Fira Math

%% set the default text family to sans-serif (Arial)
\renewcommand\familydefault{\sfdefault}

%% -------------------- Page layout and paragraphs ----------------
\RequirePackage{microtype}  % subtle justification improvements
\RequirePackage{geometry}   % mirrored margins
\geometry{
  top = 2.5cm, 
  bottom = 2.5cm,
  inner = 3.5cm,
  outer = 2.5cm,
}
\linespread{1.3} % 1.5 line spacing
\setlength{\parindent}{1.25cm} % paragraph indent
\RequirePackage{indentfirst} % indent the first paragraph
\pagestyle{plain} % page numbers in footer, plain style

%% spacing around tables and figures
\setlength{\textfloatsep}{6pt}
\setlength{\floatsep}{6pt}
\setlength{\intextsep}{6pt}

%% avoid blank pages
\makeatletter
\renewcommand{\cleardoublepage}{\clearpage}
\makeatother

%table of contents
\makeatletter
\renewcommand{\numberline}[1]{
  \hb@xt@\@tempdima{#1.\hfil} % dot after the number
}
\makeatother

%% -------------------- Heading formatting -----------------------
%% sizes and styles of titles: chapter/section/subsection/appendices
\RequirePackage{titlesec}
\titleformat{\chapter}[hang]
{\normalfont\fontsize{12}{15}\bfseries\MakeUppercase}{\thechapter.}{1em}{}
\titlespacing*{\chapter}{0pt}{12pt}{6pt}

\titleformat{\section}[hang]
{\normalfont\fontsize{10}{12}\bfseries\itshape}{\thesection.}{0.5em}{}
\titlespacing*{\section}{0pt}{12pt}{6pt}

\titleformat{\subsection}[hang]
{\normalfont\fontsize{10}{12}\itshape}{\thesubsection.}{0.5em}{}
\titlespacing*{\subsection}{0pt}{12pt}{6pt}

%% appendices
\RequirePackage[toc,page]{appendix}

%% --- Biblatex configuration with biber backend and ISO 690 style ---
\RequirePackage[
  backend=biber,        % use biber as the bibliography processing backend
  style=iso-numeric,    % numeric citation style compatible with ISO 690
  giveninits=true,      % shorten given names to initials (e.g. J., A.B.)
  sorting=none,         % do not sort the bibliography – keep citation order
  date=iso,             % use ISO date format (YYYY-MM-DD)
  urldate=iso,          % use ISO date format for access dates (urldate)
  seconds=true,         % include seconds in time fields when available
  autolang=other,       % choose language automatically based on the langid field
]{biblatex}

%% language mappings to localised strings (Polish and English variants) 
\DeclareLanguageMapping{polish}{polish-iso}
\DeclareLanguageMapping{english}{english-iso}
\DeclareLanguageMapping{american}{english-iso}
\DeclareLanguageMapping{british}{english-iso}

%% english localisation of bibliography captions and URL-related strings
\DefineBibliographyStrings{english}{
  bibliography = {Bibliography}, % title of the bibliography section
  references   = {Bibliography}, % alternative title (for \printbibliography[type=...])
  urlfrom      = {},                 % prefix printed before the URL (left empty)
  urlseen      = {date of access},     % label for the date when the online source was accessed
  online       = {online},           % label for online sources
}

%% formatting of access dates and URL fields
\DeclareFieldFormat{urldate}{(\bibstring{urlseen}\space#1)} 
\DeclareFieldFormat{url}{\bibstring{urlfrom}\addcolon\space\url{#1}}

% macro combining URL and access date in one block
\renewbibmacro*{url+urldate}{
  \printfield[url]{url}%    % print the URL field
  \setunit{\addcomma\space} % separator: comma followed by a space
  \printurldate             % print the access date
}

%% -------------------- Graphics ---------------------------------
\RequirePackage{pdfpages} % insert PDF files
\RequirePackage{float} % placement control for figures/tables
\RequirePackage{placeins}
\raggedbottom % global bottom align

%% -------------------- Tables / Lists / Mathematics --------------
\RequirePackage{multirow} % merge cells across rows
\RequirePackage{tabularx,array} % automatic column widths

\RequirePackage{enumerate} % custom numbering
\RequirePackage{enumitem} % full control of lists

\RequirePackage{amsmath} % advanced mathematics typesetting
\RequirePackage{amsfonts} % math fonts/style

%% figure and table captions
\RequirePackage[font=small,labelfont=bf]{caption}
\captionsetup{
	labelsep=period,
	figurename=Fig.,
	tablename=Table.
}

%% -------------------- Source code (listings) --------------------
\RequirePackage{xcolor}
\setmonofont{DejaVu Sans Mono}[Scale=MatchLowercase]
\setminted{
    frame=single,
    fontsize=\small,
    breaklines=true,
    numbers=left,
    numbersep=5pt
}

%% -------------------- Macros -----------------------------------
%% defining new commands and environments
\RequirePackage{xparse}
\RequirePackage{etoolbox}

%% ------------- Hyperlinks (place at the end!) -------------------
\PassOptionsToPackage{hyphens}{url}
\RequirePackage[hidelinks]{hyperref}
\hypersetup{
	colorlinks=true,
	linkcolor=black,
	anchorcolor=black,
	citecolor=black,
	menucolor=black,
	filecolor=black, 
	runcolor =black, 
	urlcolor=blue,
}
\urlstyle{same} % URLs styled like normal text
