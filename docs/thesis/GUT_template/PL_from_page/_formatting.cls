%% ============================================================
%% Ustawienia formatowania dokumentu
%% Zgodność z wytycznymi z ZR PG 45-2024
%% Wymagane pakiety
%% ============================================================

%% -------------------- Klasa bazowa _formatting  -------------------
\ProvidesClass{_formatting}
%% A4, rozmiar czcionki 10pt, druk dwustronny
\LoadClass[10pt,a4paper,twoside]{report}

%% -------------------- Język, czcionki i skład ----------------------
\RequirePackage{fontspec}
\defaultfontfeatures{Ligatures=TeX}
\RequirePackage{polyglossia}
\setdefaultlanguage[vlna=true]{polish}
\setotherlanguage{english}
\RequirePackage[newfloat]{minted}
\RequirePackage{csquotes}
\defaultfontfeatures{Ligatures=TeX, Scale=MatchLowercase}

%% użycie kroju Arial z lokalnego folderu fonts/
\setmainfont{Arial}[
  Path=fonts/,
  UprightFont    = arial.ttf,
  BoldFont       = arialbd.ttf,
  ItalicFont     = ariali.ttf,
  BoldItalicFont = arialbi.ttf,
  Scale=1.0
]

%% ta sama rodzina (Arial) ustawiona jako krój bezszeryfowy
\setsansfont{Arial}[
  Path=fonts/,
  UprightFont    = arial.ttf,
  BoldFont       = arialbd.ttf,
  ItalicFont     = ariali.ttf,
  BoldItalicFont = arialbi.ttf,
  Scale=1.0
]

%% matematyka: osobny krój, ponieważ Arial nie zawiera glifów matematycznych
\RequirePackage{unicode-math}
\setmathfont{Latin Modern Math} % ewentualnie STIX Two Math / XITS / Fira Math

%% domyślną rodziną tekstu ustaw bezszeryfową (Arial)
\renewcommand\familydefault{\sfdefault}

%% -------------------- Układ strony i akapitu ---------------------
\RequirePackage{microtype}  % subtelne poprawki justowania
\RequirePackage{geometry}   % marginesy lustrzane
\geometry{
  top = 2.5cm, 
  bottom = 2.5cm,
  inner = 3.5cm,
  outer = 2.5cm,
}
\linespread{1.3} % interlinia 1.5 wiersza;
\setlength{\parindent}{1.25cm} % wcięcie akapitu
\RequirePackage{indentfirst} % wcięcie pierwszego akapitu
\pagestyle{plain} % numeracja w stopce, styl plain

%% odstępy dla tabel i rysunków
\setlength{\textfloatsep}{6pt}
\setlength{\floatsep}{6pt}
\setlength{\intextsep}{6pt}

%% unikanie pustych stron 
\makeatletter
\renewcommand{\cleardoublepage}{\clearpage}
\makeatother

%spis treści
\makeatletter
\renewcommand{\numberline}[1]{
  \hb@xt@\@tempdima{#1.\hfil} % kropkę po numerze
}
\makeatother

%% ---------- Konfiguracja biblatex z backendem biber i stylem ISO 690 -----
\RequirePackage[
  backend=biber,        % użycie programu biber do przetwarzania bibliografii
  style=iso-numeric,    % styl cytowań i bibliografii: numeryczny, zgodny z ISO 690
  giveninits=true,      % imiona autorów skracane do inicjałów (np. J., A.B.)
  sorting=none,         % brak sortowania – pozycje w bibliografii w kolejności cytowania
  date=iso,             % zapis dat w formacie ISO (rrrr-mm-dd)
  urldate=iso,          % zapis dat dostępu (urldate) w formacie ISO
  seconds=true,         % uwzględnianie sekund w polu czasu, jeśli występują
  autolang=other,       % automatyczny dobór języka na podstawie pola langid
]{biblatex}

%% mapowanie języków na pliki definiujące lokalne napisy (polskie i angielskie)
\DeclareLanguageMapping{polish}{polish-iso}
\DeclareLanguageMapping{english}{english-iso}
\DeclareLanguageMapping{american}{english-iso}
\DeclareLanguageMapping{british}{english-iso}

%% polskie napisy w bibliografii i przy adresach URL
\DefineBibliographyStrings{polish}{
  bibliography = {Wykaz literatury}, % tytuł działu bibliografii
  references   = {Wykaz literatury}, % alternatywny tytuł (dla \printbibliography[type=...])
  urlfrom      = {},                 % poprzedzający tekst przy adresie URL
  urlseen      = {data dostępu},     % opis daty dostępu do źródła online
  online       = {online},           % oznaczenie źródeł elektronicznych
}

%% formatowanie daty dostępu i adresu URL
\DeclareFieldFormat{urldate}{(\bibstring{urlseen}\space#1)}
\DeclareFieldFormat{url}{\bibstring{urlfrom}\addcolon\space\url{#1}}

%% połączenie adresu URL z datą dostępu w jednym makrze
\renewbibmacro*{url+urldate}{
  \printfield[url]{url}    % wydruk adresu URL
  \setunit{\addcomma\space}% separator: przecinek i spacja
  \printurldate            % wydruk daty dostępu
}

%% -------------------- Formatowanie nagłówków ---------------------
%% rozmiary i styl tytułów: rozdział/sekcja/podsekcja/dodatki
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

%% dodatki
\RequirePackage[toc,page]{appendix}

%% -------------------- Grafika  --------------------------------
\RequirePackage{pdfpages} % wstawianie plików PDF
\RequirePackage{float} % kontrola położenia rys./tab.
\RequirePackage{placeins}
\raggedbottom %globalnie wyrównywanie do dołu

%% -------------------- Tabele / Listy / Matematyka -------------
\RequirePackage{multirow} % scalanie komórek w wierszach 
\RequirePackage{tabularx,array} % automatyczna szerokość kolumn

\RequirePackage{enumerate} % niestandardowe numerowanie
\RequirePackage{enumitem} % pełna kontrola list

\RequirePackage{amsmath} % dodatkowe wyrażanie matematyczne
\RequirePackage{amsfonts} % fonty/styl matematyczny

%% podpisy rysunków i tabel
\RequirePackage[font=small,labelfont=bf]{caption}
\captionsetup{
	labelsep=period,
	figurename=Rys.,
	tablename=Tabela.
}

%% -------------------- Kod źródłowy (listings) ---------------
\RequirePackage{xcolor}
\setmonofont{DejaVu Sans Mono}[Scale=MatchLowercase]
\setminted{
    frame=single,
    fontsize=\small,
    breaklines=true,
    numbers=left,
    numbersep=5pt
}

%% -------------------- Makra ------------------------------
%% definiowanie nowych komend oraz środowisk
\RequirePackage{xparse}
\RequirePackage{etoolbox}

%% ------------- Hiperłącza (umieszczamy na końcu!) --------
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
\urlstyle{same} %url jako zwykły test
