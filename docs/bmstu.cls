%% bmstu.cls
%% Copyright 2022 Mikael Novikov
%
% This work may be distributed and/or modified under the
% conditions of the LaTeX Project Public License, either version 1.3
% of this license or (at your option) any later version.
% The latest version of this license is in
%   http://www.latex-project.org/lppl.txt
% and version 1.3 or later is part of all distributions of LaTeX
% version 2005/12/01 or later.
%
% This work has the LPPL maintenance status `maintained'.
%
% The Current Maintainer of this work is Mikael Novikov.
%
% This work consists of all files listed in manifest.txt.

\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{bmstu}[2022/01/24 v1.0.0 BMSTU class]

% Установка базового класса
\LoadClass[a4paper, oneside, 14pt]{extreport}

% Настройка кодировки и языков
\RequirePackage[T1,T2A]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[english,main=russian]{babel}
% \RequirePackage{fix-cm}

% Настройка полей
\RequirePackage[
        left=30mm,
        right=10mm,
        top=20mm,
        bottom=20mm,
]{geometry}

% Настройка переносов
\RequirePackage{microtype}
\sloppy

% Настройка межстрочного интервала
\RequirePackage{setspace}
\onehalfspacing

% Настройка абзацного отступа
\RequirePackage{indentfirst}
\setlength{\parindent}{12.5mm}

% Настройка заголовков
\makeatletter
\renewcommand\LARGE{\@setfontsize\LARGE{22pt}{20}}
\renewcommand\Large{\@setfontsize\Large{20pt}{20}}
\renewcommand\large{\@setfontsize\large{16pt}{20}}
\makeatother
\RequirePackage{titlesec}
\titleformat{\chapter}{\large\bfseries}{\thechapter}{14pt}{\large\bfseries}
\titleformat{name=\chapter,numberless}{}{}{0pt}{\large\bfseries\centering}
\titleformat{\section}{\large\bfseries}{\thesection}{14pt}{\large\bfseries}
\titleformat{\subsection}{\large\bfseries}{\thesubsection}{14pt}{\large\bfseries}
\titlespacing{\chapter}{12.5mm}{-22pt}{10pt}
\titlespacing{\section}{12.5mm}{10pt}{10pt}
\titlespacing{\subsection}{12.5mm}{10pt}{10pt}

% Подключение пакета цветов
\RequirePackage{xcolor}

\RequirePackage{bmstu/sty/bmstu-figure}
\RequirePackage{bmstu/sty/bmstu-listing}

% Подключение пакетов таблиц
\RequirePackage{tabularx}
\RequirePackage{booktabs}

% Настройка подписей
\RequirePackage[
	labelsep=endash,
	figurename=Рисунок,
	singlelinecheck=false,
]{caption}
\captionsetup[figure]{justification=centering}

% Подключение пакетов для альбомной ориентации
\RequirePackage{lscape}
\RequirePackage{afterpage}

% Подключение математических пакетов
\RequirePackage{amsmath}
\RequirePackage{amssymb}

% Настройка библиографии
\RequirePackage[
	style=gost-numeric,
	language=auto,
	autolang=other,
	sorting=none,
 	backend=biber,
]{biblatex}
\RequirePackage{csquotes}
\DeclareFieldFormat{urldate}{(дата обращения:\addspace\thefield{urlday}\adddot \thefield{urlmonth}\adddot\thefield{urlyear})}

% Подключение пакета гиперссылок
\RequirePackage[unicode,hidelinks]{hyperref}

% Подключение пакета управляющих конструкций
\RequirePackage{xifthen}

\RequirePackage{bmstu/sty/bmstu-title}
\RequirePackage{bmstu/sty/bmstu-essay}
\RequirePackage{bmstu/sty/bmstu-toc}
\RequirePackage{bmstu/sty/bmstu-defabbr}
\RequirePackage{bmstu/sty/bmstu-biblio}
\RequirePackage{bmstu/sty/bmstu-appendix}

\RequirePackage{svg}

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codeblue}{rgb}{0.11,0.49,0.83}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}
\definecolor{black}{rgb}{0.0,0.0,0.0}

\RequirePackage{listingsutf8}

\lstdefinelanguage{docker}{
  keywords={FROM, RUN, COPY, ADD, ENTRYPOINT, CMD,  ENV, ARG, WORKDIR, EXPOSE, LABEL, USER, VOLUME, STOPSIGNAL, ONBUILD, MAINTAINER},
  keywordstyle=\color{codeblue}\bfseries,
  identifierstyle=\color{black},
  sensitive=false,
  comment=[l]{\#},
  commentstyle=\color{codegreen}\ttfamily,
  stringstyle=\color{codegreen}\ttfamily,
  morestring=[b]',
  morestring=[b]"
}


\lstdefinelanguage{docker-compose-2}{
  keywords={version, services},
  keywordstyle=\color{codeblue}\bfseries,
  keywords=[2]{image, environment, ports, container_name, ports, links, build, depends_on, restart, expose, command, volumes, networks},
  keywordstyle=[2]\color{red}\bfseries,
  identifierstyle=\color{black},
  sensitive=false,
  comment=[l]{\#},
  commentstyle=\color{codeblue}\ttfamily,
  stringstyle=\color{codegreen}\ttfamily,
  morestring=[b]',
  morestring=[b]"
}


\lstdefinelanguage{actions}{
  keywords={name, on, jobs},
  keywordstyle=\color{codeblue}\bfseries,
  keywords=[2]{needs, with, run, uses, file, context, tags, username, password, host, key, port, source, overwrite, target, strip_components, tar_tmp_path},
  keywordstyle=[2]\color{red}\bfseries,
  identifierstyle=\color{black},
  sensitive=false,
  comment=[l]{\#},
  commentstyle=\color{codeblue}\ttfamily,
  stringstyle=\color{codegreen}\ttfamily,
  morestring=[b]',
  morestring=[b]"
}

\lstdefinelanguage{nginx}{
  keywords={upstream, server},
  keywordstyle=\color{codeblue}\bfseries,
  keywords=[2]{listen, server_name, include, location},
  keywordstyle=[2]\color{red}\bfseries,
  identifierstyle=\color{black},
  sensitive=false,
  comment=[l]{\#},
  commentstyle=\color{codeblue}\ttfamily,
  stringstyle=\color{codegreen}\ttfamily,
  morestring=[b]',
  morestring=[b]"
}


% Создание листинга
\newcommand{\listingfile}[5]{
\lstinputlisting[
    basicstyle=\small,
    float={#5},
    frame=single,
    abovecaptionskip=-5pt,
    identifierstyle=\color{black},
    commentstyle=\color{codegreen}\ttfamily,
    keywordstyle=\color{codeblue}\bfseries,
    stringstyle=\color{codegreen}\ttfamily,
    showspaces=false,
    showstringspaces=false,
    caption={#4},
    label={lst: #2},
    language={#3},
]{inc/lst/#1}
}

\newcommand{\listingfileX}[5]{
\lstinputlisting[
    basicstyle=\small,
    frame=single,
    abovecaptionskip=-5pt,
    identifierstyle=\color{black},
    commentstyle=\color{codegreen}\ttfamily,
    keywordstyle=\color{codeblue}\bfseries,
    stringstyle=\color{codegreen}\ttfamily,
    showspaces=false,
    showstringspaces=false,
    caption={#4},
    label={lst: #2},
    language={#3},
    #5,
]{inc/lst/#1}
}

\newcommand{\code}[1]{\texttt{#1}}

\usepackage{listings}

\usepackage{subcaption}
\captionsetup[subfigure]{labelformat=empty}

\usepackage{svg}

% Значения по умолчанию
\lstset{
  basicstyle= \footnotesize,
  breakatwhitespace=true,% разрыв строк только на whitespacce
  breaklines=true,       % переносить длинные строки
%   captionpos=b,          % подписи снизу -- вроде не надо
  inputencoding=koi8-r,
  numbers=left,          % нумерация слева
  numberstyle=\footnotesize,
  showspaces=false,      % показывать пробелы подчеркиваниями -- идиотизм 70-х годов
  showstringspaces=false,
  showtabs=false,        % и табы тоже
  stepnumber=1,
  tabsize=4,              % кому нужны табы по 8 символов?
  frame=single,
  xleftmargin=2.4em,
  framexleftmargin=2em
}