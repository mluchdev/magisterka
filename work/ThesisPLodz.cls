\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ThesisPLodz}[2026/03/10 Praca Dyplomowa Politechnika Łódzka]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wybór loga wydziału
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\@logoimage}{logo/PL.pdf}
\DeclareOption{W1}{
    \renewcommand{\@logoimage}{logo/PL_W1_M.pdf}
}
\DeclareOption{W2}{
    \renewcommand{\@logoimage}{logo/PL_W2_EEIA.pdf}
}
\DeclareOption{W3}{
    \renewcommand{\@logoimage}{logo/PL_W3_Ch.pdf}
}
\DeclareOption{W4}{
    \renewcommand{\@logoimage}{logo/PL_W4_WiW.png}
}
\DeclareOption{W5}{
    \renewcommand{\@logoimage}{logo/PL_W5_BiNoZ.pdf}
}
\DeclareOption{W6}{
    \renewcommand{\@logoimage}{logo/PL_W6_BAIS.pdf}
}
\DeclareOption{W7}{
    \renewcommand{\@logoimage}{logo/PL_W7_FTIMS.pdf}
}
\DeclareOption{W8}{
    \renewcommand{\@logoimage}{logo/PL_W8_OiZ.pdf}
}
\DeclareOption{W9}{
    \renewcommand{\@logoimage}{logo/PL_W9_IPOS.pdf}
}
\DeclareOption{IFE}{
    \renewcommand{\@logoimage}{logo/PL_WX_IFE.pdf}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wybór języka pracy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\@babellanguages}{english,polish}
\DeclareOption{polski}{\renewcommand{\@babellanguages}{english,main=polish}}
\DeclareOption{english}{\renewcommand{\@babellanguages}{polish,main=english}}
\DeclareOption{francaise}{\renewcommand{\@babellanguages}{english,main=french}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Poufność pracy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newif\if@confidential
\@confidentialfalse
\DeclareOption{confidential}{\@confidentialtrue}
\newcommand{\@copynumber}{1}
\newcommand{\copynumber}[1]{\renewcommand{\@copynumber}{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Przetworzenie zadanej konfiguracji
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ProcessOptions\relax

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wczytanie klasy bazowej
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\LoadClass[a4paper, 12pt, titlepage, leqno, twoside, openright]{book}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wymiary strony
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[
    twoside=true,
    left=3cm, % inner
    right=2cm, % outer
    top=2.5cm,
    bottom=2.5cm,
    headheight=1.25cm,
    footskip=1.25cm,
]{geometry}
\RequirePackage[onehalfspacing]{setspace}
\RequirePackage[skip=4pt]{parskip}
\raggedbottom

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Czcionka - kodowanie i symbole
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}

\RequirePackage{amsfonts}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{bm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ustawienie języka i zmiana nazw figur i tabel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[\@babellanguages]{babel}
\addto\captionspolish{\renewcommand{\figurename}{Rys.}}
\addto\captionspolish{\renewcommand{\tablename}{Tabela}}
\addto\captionspolish{\renewcommand{\lstlistingname}{Kod źródłowy}}
\addto\captionspolish{\renewcommand{\lstlistlistingname}{Spis kodów źródłowych}}

\addto\captionsenglish{\renewcommand{\lstlistlistingname}{List of Listings}}

\addto\captionsfrench{\renewcommand{\lstlistingname}{Code source}}
\addto\captionsfrench{\renewcommand{\lstlistlistingname}{Liste des codes sources}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Czcionka - Sans Serif zgodnie z Zarządzeniem 6/2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[scaled=0.95]{helvet}
\renewcommand{\familydefault}{\sfdefault}

\RequirePackage{etoolbox}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Oznakowanie pracy poufnej
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\if@confidential
    \RequirePackage{draftwatermark}
    \SetWatermarkText{TAJEMNICA P\L{} Egz. nr \@copynumber}
    \SetWatermarkFontSize{12pt}
    \SetWatermarkAngle{0}
    \SetWatermarkColor[gray]{0}
    \SetWatermarkHorCenter{\dimexpr\paperwidth-3cm\relax}
    \SetWatermarkVerCenter{1cm}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Komendy do ustawienia danych autora i pracy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\@authortitle}{}
\newcommand{\@authorname}{}
\newcommand{\@thesistitle}{}
\newcommand{\@thesisshorttitle}{}
\newcommand{\@studentnumber}{}
\newcommand{\@degreename}{}
\newcommand{\@degreename@magisterska}{magisterska}
\newcommand{\@degreename@inzynierska}{inżynierska}
\newcommand{\@degreename@licencjacka}{licencjacka}
\newcommand{\@fieldofstudy}{}
\newcommand{\@department}{}
\newcommand{\@supervisor}{}
\newcommand{\@cosupervisor}{}
\newcommand{\@partnersupervisor}{}

\newcommand{\@abstractcontent}{}
\newcommand{\@keywordscontent}{}
\newcommand{\@abstractsecondarycontent}{}
\newcommand{\@keywordssecondarycontent}{}
\newcommand{\@bibliographyfile}{}

\newcommand{\authortitle}[1]{\renewcommand{\@authortitle}{#1}}
\newcommand{\authorname}[1]{\renewcommand{\@authorname}{#1}}
\newcommand{\thesistitle}[1]{\renewcommand{\@thesistitle}{#1}}
\newcommand{\thesisshorttitle}[1]{\renewcommand{\@thesisshorttitle}{#1}}
\newcommand{\studentnumber}[1]{\renewcommand{\@studentnumber}{#1}}
\newcommand{\abstract}[1]{\renewcommand{\@abstractcontent}{#1}}
\newcommand{\keywords}[1]{\renewcommand{\@keywordscontent}{#1}}
\newcommand{\abstractsecondary}[1]{\renewcommand{\@abstractsecondarycontent}{#1}}
\newcommand{\keywordssecondary}[1]{\renewcommand{\@keywordssecondarycontent}{#1}}
\newcommand{\degreename}[1]{%
    \expandafter\ifstrequal\expandafter{#1}{magisterska}{\renewcommand{\@degreename}{\@degreename@magisterska}}{%
    \expandafter\ifstrequal\expandafter{#1}{inzynierska}{\renewcommand{\@degreename}{\@degreename@inzynierska}}{%
    \expandafter\ifstrequal\expandafter{#1}{licencjacka}{\renewcommand{\@degreename}{\@degreename@licencjacka}}{%
    \ClassError{ThesisPLodz}{Unknown degree name '#1'. Use: magisterska, inzynierska, licencjacka}{}%
    }}}%
}
\newcommand{\fieldofstudy}[1]{\renewcommand{\@fieldofstudy}{#1}}
\newcommand{\department}[1]{\renewcommand{\@department}{#1}}
\newcommand{\supervisor}[1]{\renewcommand{\@supervisor}{#1}}
\newcommand{\cosupervisor}[1]{\renewcommand{\@cosupervisor}{#1}}
\newcommand{\partnersupervisor}[1]{\renewcommand{\@partnersupervisor}{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Napisy zależne od języka
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\@thesislabel}{PRACA DYPLOMOWA}
\newcommand{\@fieldprefix}{na kierunku}
\newcommand{\@supervisorlabel}{Promotor}
\newcommand{\@cosupervisorlabel}{Opiekun pomocniczy}
\newcommand{\@partnersupervisorlabel}{Promotor z uczelni partnerskiej}
\newcommand{\@blankpagelabel}{Ta strona celowo pozostawiona pusta}
\newcommand{\@symbolslistlabel}{Wykaz symboli i oznaczeń}
\newcommand{\@abbreviationslistlabel}{Wykaz używanych skrótów}
\newcommand{\@definitionslistlabel}{Lista definicji}
\newcommand{\@attachmentslistlabel}{Wykaz załączników}
\newcommand{\@attachmentslabel}{Załączniki}
\newcommand{\@abstractlabel}{Streszczenie}
\newcommand{\@keywordslabel}{Słowa kluczowe}
\newcommand{\@abstractsecondarylabel}{Abstract}
\newcommand{\@keywordssecondarylabel}{Keywords}
\addto\captionspolish{%
    \renewcommand{\@thesislabel}{PRACA DYPLOMOWA}%
    \renewcommand{\@fieldprefix}{na kierunku}%
    \renewcommand{\@supervisorlabel}{Promotor}%
    \renewcommand{\@cosupervisorlabel}{Promotor pomocniczy}%
    \renewcommand{\@partnersupervisorlabel}{Promotor z uczelni partnerskiej}%
    \renewcommand{\@blankpagelabel}{Ta strona celowo pozostawiona pusta}%
    \renewcommand{\@symbolslistlabel}{Wykaz symboli i oznaczeń}%
    \renewcommand{\@abbreviationslistlabel}{Wykaz używanych skrótów}%
    \renewcommand{\@definitionslistlabel}{Lista definicji}%
    \renewcommand{\@attachmentslistlabel}{Wykaz załączników}%
    \renewcommand{\@attachmentslabel}{Załączniki}%
    \renewcommand{\@abstractlabel}{Streszczenie}%
    \renewcommand{\@keywordslabel}{Słowa kluczowe}%
    \renewcommand{\@abstractsecondarylabel}{Abstract}%
    \renewcommand{\@keywordssecondarylabel}{Keywords}%
    \renewcommand{\@degreename@magisterska}{magisterska}%
    \renewcommand{\@degreename@inzynierska}{inżynierska}%
    \renewcommand{\@degreename@licencjacka}{licencjacka}%
}
\addto\captionsenglish{%
    \renewcommand{\@thesislabel}{DIPLOMA THESIS}%
    \renewcommand{\@fieldprefix}{in the field of}%
    \renewcommand{\@supervisorlabel}{Supervisor}%
    \renewcommand{\@cosupervisorlabel}{Co-supervisor}%
    \renewcommand{\@partnersupervisorlabel}{Partner university supervisor}%
    \renewcommand{\@blankpagelabel}{This page intentionally left blank}%
    \renewcommand{\@symbolslistlabel}{List of Symbols}%
    \renewcommand{\@abbreviationslistlabel}{List of Abbreviations}%
    \renewcommand{\@definitionslistlabel}{List of Definitions}%
    \renewcommand{\@attachmentslistlabel}{List of Attachments}%
    \renewcommand{\@attachmentslabel}{Attachments}%
    \renewcommand{\@abstractlabel}{Abstract}%
    \renewcommand{\@keywordslabel}{Keywords}%
    \renewcommand{\@abstractsecondarylabel}{Streszczenie}%
    \renewcommand{\@keywordssecondarylabel}{S\l{}owa kluczowe}%
    \renewcommand{\@degreename@magisterska}{master's}%
    \renewcommand{\@degreename@inzynierska}{engineer's}%
    \renewcommand{\@degreename@licencjacka}{bachelor's}%
}
\addto\captionsfrench{%
    \renewcommand{\@thesislabel}{TH\`ESE DE DIPL\^OME}%
    \renewcommand{\@fieldprefix}{dans le domaine de}%
    \renewcommand{\@supervisorlabel}{Directeur de th\`ese}%
    \renewcommand{\@cosupervisorlabel}{Co-directeur}%
    \renewcommand{\@partnersupervisorlabel}{Directeur de l'universit\'e partenaire}%
    \renewcommand{\@blankpagelabel}{Cette page est laiss\'ee intentionnellement vide}%
    \renewcommand{\@symbolslistlabel}{Liste des symboles}%
    \renewcommand{\@abbreviationslistlabel}{Liste des abr\'eviations}%
    \renewcommand{\@definitionslistlabel}{Liste des d\'efinitions}%
    \renewcommand{\@attachmentslistlabel}{Liste des annexes}%
    \renewcommand{\@attachmentslabel}{Annexes}%
    \renewcommand{\@abstractlabel}{R\'esum\'e}%
    \renewcommand{\@keywordslabel}{Mots cl\'es}%
    \renewcommand{\@abstractsecondarylabel}{Abstract}%
    \renewcommand{\@keywordssecondarylabel}{Keywords}%
    \renewcommand{\@degreename@magisterska}{de master}%
    \renewcommand{\@degreename@inzynierska}{d'ing\'enieur}%
    \renewcommand{\@degreename@licencjacka}{de licence}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nagłówki i numeracja stron
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{fancyhdr}

\fancypagestyle{thesis}{%
    \fancyhf{}%
    \fancyhead[RO,RE]{\@thesisshorttitle}%
    \fancyhead[LO,LE]{\@authortitle~\@authorname}%
    \fancyfoot[RO,LE]{\fontsize{12}{14}\selectfont\thepage}%
    \renewcommand{\headrulewidth}{0.4pt}%
    \renewcommand{\footrulewidth}{0pt}%
}

\fancypagestyle{plain}{%
    \fancyhf{}%
    \fancyhead[RO,RE]{\@thesisshorttitle}%
    \fancyhead[LO,LE]{\@authortitle~\@authorname}%
    \fancyfoot[RO,LE]{\fontsize{12}{14}\selectfont\thepage}%
    \renewcommand{\headrulewidth}{0.4pt}%
    \renewcommand{\footrulewidth}{0pt}%
}

% Strona tytułowa bez nagłówka i numeru
\let\oldtitlepage\titlepage
\renewcommand{\titlepage}{\oldtitlepage\thispagestyle{empty}}

% Puste strony (np. po stronie tytułowej) bez nagłówka i numeru
\let\oldcleardoublepage\cleardoublepage
\renewcommand{\cleardoublepage}{%
    \clearpage
    \if@twoside
        \ifodd\c@page\else
            \thispagestyle{empty}%
            \addtocounter{page}{-1}%
            \null\vfill
            \begin{center}
                {\textit{\@blankpagelabel}}
            \end{center}
            \vfill\newpage
            \if@twocolumn\hbox{}\newpage\fi
        \fi
    \fi
}

% frontmatter: numeracja rzymska
\let\oldfrontmatter\frontmatter
\renewcommand{\frontmatter}{%
    \oldfrontmatter
    \pagestyle{thesis}%
}

% mainmatter: numeracja arabska
\let\oldmainmatter\mainmatter
\renewcommand{\mainmatter}{%
    \oldmainmatter
    \pagestyle{thesis}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Formatowanie tytułów rozdziałów i sekcji
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\patchcmd{\l@chapter}{\hfil}{\leaders\hbox{$\m@th\mkern 4.5mu\hbox{.}\mkern 4.5mu$}\hfill}{}{}
\RequirePackage{titlesec}

\titleformat{\chapter}{\fontsize{16}{19}\selectfont\bfseries}{\thechapter}{1em}{}
\titlespacing*{\chapter}{0pt}{0pt}{10pt}
\titleformat{\section}{\fontsize{14}{17}\selectfont\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\fontsize{13}{16}\selectfont\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\fontsize{13}{16}\selectfont\bfseries}{\thesubsubsection}{1em}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grafiki i listingi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{pdfpages}
\RequirePackage{graphicx}
\RequirePackage{float}
\RequirePackage[export]{adjustbox}
\RequirePackage{caption}
\DeclareCaptionFont{10pt}{\fontsize{10}{12}\selectfont}
\captionsetup{font={10pt}}
\RequirePackage{listings}
\lstset{
  basicstyle=\ttfamily,
  numbers=left,
  numberstyle=\tiny\color{gray},
  frame=single,
  breaklines=true,
  captionpos=b,
  showstringspaces=false
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Strona tytułowa
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\maketitlepage}{%
    \begin{titlepage}
        \thispagestyle{empty}%
        \begin{flushright}
            \includegraphics[height=3cm, left]{\@logoimage}
        \end{flushright}
        \vspace{2cm}
        \begin{center}
            {\fontsize{14}{17}\selectfont \@authortitle~\@authorname}\\[4pt]
            {\fontsize{14}{14}\selectfont \@studentnumber}
        \end{center}
        \vspace{2cm}
        \begin{center}
            {\fontsize{16}{17}\selectfont\bfseries \@thesislabel}\\[4pt]
            {\fontsize{16}{14}\selectfont \MakeUppercase{\@degreename}}\\[4pt]
            {\fontsize{14}{14}\selectfont \@fieldprefix~\@fieldofstudy}
        \end{center}
        \vspace{2cm}
        \begin{center}
            {\fontsize{16}{19}\selectfont\bfseries \@thesistitle}
        \end{center}
        \vfill
        \begin{flushleft}
            \fontsize{14}{16}\selectfont
            {\@department}\\
            {\@supervisorlabel: \@supervisor}\\
            \if\relax\detokenize\expandafter{\@cosupervisor}\relax\else{\@cosupervisorlabel: \@cosupervisor}\\\fi
            \if\relax\detokenize\expandafter{\@partnersupervisor}\relax\else{\@partnersupervisorlabel: \@partnersupervisor}\\\fi
        \end{flushleft}
        \vspace{1cm}
        \begin{center}
            {\fontsize{12}{14}\selectfont \L\'{O}D\'{Z} \the\year}
        \end{center}
    \end{titlepage}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Komendy do dodawania strony tytułowej i abstractu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\AtBeginDocument{
    \maketitlepage
    \frontmatter
    \tableofcontents
    %% Streszczenie i słowa kluczowe w języku głównym
    \if\relax\detokenize\expandafter{\@abstractcontent}\relax\else
        \chapter*{\@abstractlabel}
        \@abstractcontent
        \if\relax\detokenize\expandafter{\@keywordscontent}\relax\else
            \vspace{1em}
            \par
            \noindent\textbf{\@keywordslabel:}
            \noindent\@keywordscontent
        \fi
    \fi
    %% Abstract and keywords in secondary language
    \if\relax\detokenize\expandafter{\@abstractsecondarycontent}\relax\else
        \chapter*{\@abstractsecondarylabel}
        \@abstractsecondarycontent
        \if\relax\detokenize\expandafter{\@keywordssecondarycontent}\relax\else
            \vspace{1em}
            \par
            \noindent\textbf{\@keywordssecondarylabel:}
            \noindent\@keywordssecondarycontent
        \fi
    \fi
    \mainmatter
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Komendy do dodawania wykazów symboli, skrótów i definicji
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\@symbolslistcontent}{}
\newcommand{\@abbreviationslistcontent}{}
\newcommand{\@definitionslistcontent}{}
\newcommand{\symbolslist}[1]{\renewcommand{\@symbolslistcontent}{#1}}
\newcommand{\abbreviationslist}[1]{\renewcommand{\@abbreviationslistcontent}{#1}}
\newcommand{\definitionslist}[1]{\renewcommand{\@definitionslistcontent}{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bibliografia
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\bibliographyfile}[1]{\renewcommand{\@bibliographyfile}{#1}}
\bibliographystyle{apalike}

% Pogrubienie autora pracy w bibliografii
\RequirePackage{xstring}

\newcommand{\@boldauthorstring}{}
\newcommand{\boldauthorinbib}[1]{\renewcommand{\@boldauthorstring}{#1}}
\AtBeginDocument{%
    \if\relax\detokenize\expandafter{\@boldauthorstring}\relax\else
        \let\@old@lbibitem\@lbibitem
        \def\@lbibitem[#1]#2#3\par{%
            \@old@lbibitem[#1]{#2}%
            \noexpandarg
            \expanded{%
                \noexpand\StrSubstitute
                    {\unexpanded{#3}}%
                    {\@boldauthorstring}%
                    {\noexpand\textbf{\@boldauthorstring}}%
                    [\noexpand\@tempentry]%
            }%
            \fullexpandarg
            \@tempentry\par
        }%
    \fi
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Komendy do dodawania załączników
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcounter{attachmentcounter}
\newcounter{@attachmentloop}

\newcommand{\addattachment}[2]{%
    \stepcounter{attachmentcounter}%
    \expandafter\def\csname @attachtitle\arabic{attachmentcounter}\endcsname{#1}%
    \expandafter\long\expandafter\def\csname @attachcontent\arabic{attachmentcounter}\endcsname{#2}%
    \expandafter\def\csname @attachtype\arabic{attachmentcounter}\endcsname{content}%
}

\newcommand{\addpdfattachment}[2]{%
    \stepcounter{attachmentcounter}%
    \expandafter\def\csname @attachtitle\arabic{attachmentcounter}\endcsname{#1}%
    \expandafter\def\csname @attachcontent\arabic{attachmentcounter}\endcsname{#2}%
    \expandafter\def\csname @attachtype\arabic{attachmentcounter}\endcsname{pdf}%
}

\newcommand{\@renderattachmentslist}{%
    \setcounter{@attachmentloop}{0}%
    \@whilenum\value{@attachmentloop}<\value{attachmentcounter}\do{%
        \stepcounter{@attachmentloop}%
        \par\noindent\arabic{@attachmentloop}.~\csname @attachtitle\arabic{@attachmentloop}\endcsname%
    }%
}

\newcommand{\@renderattachments}{%
    \setcounter{@attachmentloop}{0}%
    \@whilenum\value{@attachmentloop}<\value{attachmentcounter}\do{%
        \stepcounter{@attachmentloop}%
        \edef\@attachtype@current{\csname @attachtype\arabic{@attachmentloop}\endcsname}%
        \expandafter\ifstrequal\expandafter{\@attachtype@current}{pdf}{%
            \edef\@tempa{\noexpand\includepdf[pages=-]{\csname @attachcontent\arabic{@attachmentloop}\endcsname}}%
            \@tempa%
        }{%
            \section*{\arabic{@attachmentloop}.~\csname @attachtitle\arabic{@attachmentloop}\endcsname}%
            \csname @attachcontent\arabic{@attachmentloop}\endcsname%
            \clearpage%
        }%
    }%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Komendy do dodawania końcowych wykazów
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\AtEndDocument{%
    %% Bibliografia
    \if\relax\detokenize\expandafter{\@bibliographyfile}\relax\else
        \bibliography{\@bibliographyfile}
        \addcontentsline{toc}{chapter}{\bibname}
    \fi
    \backmatter
    %% Wykaz rysunków
    % \ifnum\value{figure}>0
        \clearpage
        \addcontentsline{toc}{chapter}{\listfigurename}
        \listoffigures
    % \fi
    %% Wykaz tabel
    % \ifnum\value{table}>0
        \clearpage
        \addcontentsline{toc}{chapter}{\listtablename}
        \listoftables
    % \fi
    %% Wykaz listingów
    % \ifnum\value{lstlisting}>0
        \clearpage
        \addcontentsline{toc}{chapter}{\lstlistlistingname}
        \lstlistoflistings
    % \fi
    %% Wykaz symboli i oznaczeń
    \if\relax\detokenize\expandafter{\@symbolslistcontent}\relax\else
        \chapter*{\@symbolslistlabel}
        \addcontentsline{toc}{chapter}{\@symbolslistlabel}
        \@symbolslistcontent
    \fi
    %% Wykaz używanych skrótów
    \if\relax\detokenize\expandafter{\@abbreviationslistcontent}\relax\else
        \chapter*{\@abbreviationslistlabel}
        \addcontentsline{toc}{chapter}{\@abbreviationslistlabel}
        \@abbreviationslistcontent
    \fi
    %% Lista definicji
    \if\relax\detokenize\expandafter{\@definitionslistcontent}\relax\else
        \chapter*{\@definitionslistlabel}
        \addcontentsline{toc}{chapter}{\@definitionslistlabel}
        \@definitionslistcontent
    \fi
    %% Wykaz załączników (auto-generated from \addattachment)
    \ifnum\value{attachmentcounter}>0
        \chapter*{\@attachmentslistlabel}
        \addcontentsline{toc}{chapter}{\@attachmentslistlabel}
        \@renderattachmentslist
        \chapter*{\@attachmentslabel}
        \addcontentsline{toc}{chapter}{\@attachmentslabel}
        \@renderattachments
    \fi
}